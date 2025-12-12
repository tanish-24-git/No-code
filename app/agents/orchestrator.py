"""
Orchestrator agent for executing agent DAGs (Directed Acyclic Graphs).
Handles pipeline execution, dependencies, retries, and state transitions.
"""
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel
from collections import defaultdict, deque
from app.agents.base_agent import BaseAgent
from app.utils.exceptions import OrchestrationException, ValidationException
from app.infra.queue import task_queue
from tenacity import retry, stop_after_attempt, wait_exponential


class AgentNode(BaseModel):
    """Represents a node in the agent DAG."""
    agent_name: str
    agent_class: str  # Fully qualified class name
    config: Dict[str, Any] = {}


class AgentEdge(BaseModel):
    """Represents an edge (dependency) in the agent DAG."""
    from_agent: str
    to_agent: str


class PipelineConfig(BaseModel):
    """Pipeline configuration with nodes and edges."""
    run_id: str
    nodes: List[AgentNode]
    edges: List[AgentEdge]
    global_config: Dict[str, Any] = {}


class OrchestratorAgent(BaseAgent):
    """
    Orchestrates execution of agent DAGs.
    
    Responsibilities:
    - Parse agent graph (nodes + edges)
    - Perform topological sort for execution order
    - Schedule agents to task queue
    - Handle dependencies (wait for upstream completion)
    - Retry failed agents with exponential backoff
    - Track overall pipeline state
    - Emit pipeline-level events
    """
    
    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.agent_registry: Dict[str, type[BaseAgent]] = {}
    
    def register_agent(self, agent_class: type[BaseAgent], name: str):
        """Register an agent class for orchestration."""
        self.agent_registry[name] = agent_class
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent DAG.
        
        Input:
            {
                "run_id": "abc123",
                "nodes": [
                    {"agent_name": "dataset", "agent_class": "DatasetAgent", "config": {...}},
                    {"agent_name": "validation", "agent_class": "ValidationAgent", "config": {...}}
                ],
                "edges": [
                    {"from_agent": "dataset", "to_agent": "validation"}
                ],
                "global_config": {...}
            }
        
        Output:
            {
                "pipeline_status": "completed",
                "agent_results": {
                    "dataset": {...},
                    "validation": {...}
                },
                "failed_agents": []
            }
        """
        # Validate input
        pipeline = self.validate_input(input_data, PipelineConfig)
        run_id = pipeline.run_id
        
        await self._emit_log(run_id, "INFO", "Pipeline orchestration started", 
                           node_count=len(pipeline.nodes))
        
        # Build dependency graph
        graph = self._build_graph(pipeline.nodes, pipeline.edges)
        
        # Topological sort to get execution order
        execution_order = self._topological_sort(graph, pipeline.nodes)
        
        if not execution_order:
            raise OrchestrationException(
                "Failed to create execution order. Graph may contain cycles.",
                metadata={"nodes": [n.agent_name for n in pipeline.nodes]}
            )
        
        await self._emit_log(run_id, "INFO", "Execution order determined",
                           order=[n.agent_name for n in execution_order])
        
        # Execute agents in order
        agent_results = {}
        failed_agents = []
        
        for node in execution_order:
            try:
                # Prepare input for agent
                agent_input = {
                    "run_id": run_id,
                    **pipeline.global_config,
                    **node.config
                }
                
                # Add outputs from upstream agents
                upstream_agents = graph.get(node.agent_name, {}).get("dependencies", [])
                for upstream in upstream_agents:
                    if upstream in agent_results:
                        agent_input[f"{upstream}_output"] = agent_results[upstream]
                
                # Execute agent
                await self._emit_log(run_id, "INFO", f"Executing agent: {node.agent_name}")
                
                result = await self._execute_agent(node, agent_input)
                
                if result["success"]:
                    agent_results[node.agent_name] = result["data"]
                    await self._emit_log(run_id, "INFO", f"Agent completed: {node.agent_name}")
                else:
                    failed_agents.append({
                        "agent": node.agent_name,
                        "error": result["error"]
                    })
                    await self._emit_log(run_id, "ERROR", f"Agent failed: {node.agent_name}",
                                       error=result["error"])
                    
                    # Stop pipeline on failure (can be made configurable)
                    break
            
            except Exception as e:
                failed_agents.append({
                    "agent": node.agent_name,
                    "error": str(e)
                })
                await self._emit_log(run_id, "ERROR", f"Agent execution error: {node.agent_name}",
                                   error=str(e))
                break
        
        # Determine pipeline status
        pipeline_status = "completed" if not failed_agents else "failed"
        
        await self._emit_log(run_id, "INFO", f"Pipeline {pipeline_status}",
                           completed_agents=len(agent_results),
                           failed_agents=len(failed_agents))
        
        return {
            "pipeline_status": pipeline_status,
            "agent_results": agent_results,
            "failed_agents": failed_agents
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _execute_agent(self, node: AgentNode, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single agent with retry logic.
        
        Args:
            node: Agent node configuration
            agent_input: Input data for agent
        
        Returns:
            Agent output
        """
        # Get agent class from registry
        agent_class = self.agent_registry.get(node.agent_class)
        
        if not agent_class:
            raise OrchestrationException(
                f"Agent class not registered: {node.agent_class}",
                metadata={"agent_name": node.agent_name}
            )
        
        # Instantiate and run agent
        agent = agent_class()
        result = await agent.run(agent_input)
        
        return result
    
    def _build_graph(
        self,
        nodes: List[AgentNode],
        edges: List[AgentEdge]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build dependency graph from nodes and edges.
        
        Returns:
            {
                "agent_name": {
                    "node": AgentNode,
                    "dependencies": ["upstream1", "upstream2"],
                    "dependents": ["downstream1", "downstream2"]
                }
            }
        """
        graph = {}
        
        # Initialize nodes
        for node in nodes:
            graph[node.agent_name] = {
                "node": node,
                "dependencies": [],
                "dependents": []
            }
        
        # Add edges
        for edge in edges:
            if edge.from_agent not in graph or edge.to_agent not in graph:
                raise ValidationException(
                    f"Invalid edge: {edge.from_agent} -> {edge.to_agent}",
                    metadata={"edge": edge.dict()}
                )
            
            graph[edge.to_agent]["dependencies"].append(edge.from_agent)
            graph[edge.from_agent]["dependents"].append(edge.to_agent)
        
        return graph
    
    def _topological_sort(
        self,
        graph: Dict[str, Dict[str, Any]],
        nodes: List[AgentNode]
    ) -> Optional[List[AgentNode]]:
        """
        Perform topological sort using Kahn's algorithm.
        
        Returns:
            Ordered list of nodes, or None if graph has cycles
        """
        # Calculate in-degree for each node
        in_degree = {name: len(data["dependencies"]) for name, data in graph.items()}
        
        # Queue of nodes with no dependencies
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        
        result = []
        
        while queue:
            # Process node with no dependencies
            current = queue.popleft()
            result.append(graph[current]["node"])
            
            # Reduce in-degree of dependents
            for dependent in graph[current]["dependents"]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check if all nodes were processed (no cycles)
        if len(result) != len(nodes):
            return None
        
        return result


# Global orchestrator instance
orchestrator = OrchestratorAgent()
