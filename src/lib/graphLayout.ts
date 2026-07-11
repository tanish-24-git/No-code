import dagre from 'dagre';
import { Position, type Edge, type Node } from 'reactflow';

export const AGENT_NODE_W = 210;
export const AGENT_NODE_H = 76;
export const ARTIFACT_NODE_W = 160;
export const ARTIFACT_NODE_H = 44;

/**
 * dagre auto-layout, re-run on every graph mutation (ReactFlow never
 * re-layouts by itself). Top-to-bottom: orchestrator on top, workers below,
 * artifacts hanging off their producers.
 */
export function layoutGraph(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', ranksep: 70, nodesep: 36 });
  for (const n of nodes) {
    const artifact = n.type === 'artifact';
    g.setNode(n.id, {
      width: artifact ? ARTIFACT_NODE_W : AGENT_NODE_W,
      height: artifact ? ARTIFACT_NODE_H : AGENT_NODE_H,
    });
  }
  for (const e of edges) {
    if (g.hasNode(e.source) && g.hasNode(e.target)) g.setEdge(e.source, e.target);
  }
  dagre.layout(g);
  return nodes.map((n) => {
    const pos = g.node(n.id);
    const artifact = n.type === 'artifact';
    const w = artifact ? ARTIFACT_NODE_W : AGENT_NODE_W;
    const h = artifact ? ARTIFACT_NODE_H : AGENT_NODE_H;
    return {
      ...n,
      position: { x: (pos?.x ?? 0) - w / 2, y: (pos?.y ?? 0) - h / 2 },
      targetPosition: Position.Top,
      sourcePosition: Position.Bottom,
    };
  });
}
