/**
 * Frontend-side mirror of backend Pydantic schemas. Keep these in sync with
 * `backend/app/api/schemas/*` when the backend changes.
 */

// ── Health ────────────────────────────────────────────────────────────────

export type Health = {
  status: string;
  hardware?: { device?: string; gpu_name?: string | null; vram_gb?: number | null };
  llm: { configured: boolean; provider?: string | null; model?: string | null };
};

// ── Settings ──────────────────────────────────────────────────────────────

export type Provider = string;

export type ProviderInfo = {
  name: Provider;
  label: string;
  engine: 'anthropic' | 'openai';
  base_url: string;
  needs_key: boolean;
  sample_models: string[];
  notes?: string;
};

export type Settings = {
  llm_provider: Provider | null;
  llm_model: string;
  llm_base_url: string;
  llm_api_key_masked: string | null;
  llm_api_key_set: boolean;
  llm_source: 'env' | 'ui' | 'unset';
  hf_token_masked: string | null;
  hf_token_set: boolean;
  hf_username: string | null;
  hf_source: 'env' | 'ui' | 'unset';
  auto_config_on_upload: boolean;
  show_agent_reasoning: boolean;
  is_configured: boolean;
};

// ── Datasets ──────────────────────────────────────────────────────────────

export type Dataset = {
  id: string;
  name: string;
  file_path: string;
  file_type: string;
  row_count: number;
  column_names: string[];
  column_types: Record<string, string>;
  sample_rows: Record<string, unknown>[];
  size_bytes: number;
  is_analyzed: boolean;
  analysis: Record<string, unknown> | null;
  created_at: string;
};

export type DatasetUploadResponse = {
  dataset: Dataset;
  session_id: string;
  llm_probe: LLMProbe;
};

export const ACCEPTED_DATASET_TYPES =
  '.csv,.json,.jsonl,.txt,.md,.pdf,.docx,.html';

export type LLMProbe = {
  ok: boolean;
  mode: 'full_agent' | 'deterministic' | string;
  provider: string | null;
  model: string | null;
  latency_ms: number;
  detail: string;
};

// ── Pipelines ─────────────────────────────────────────────────────────────

export type NodeGraph = {
  nodes: { id: string; type: string; position: { x: number; y: number }; data: Record<string, unknown> }[];
  edges: { id: string; source: string; target: string }[];
  viewport: { x: number; y: number; zoom: number };
};

export type PipelineConfig = {
  project_name: string;
  description?: string | null;
  tags: string[];
  dataset_id?: string | null;
  target_column?: string | null;
  input_columns: string[];
  split_ratio: number;
  task_type: 'Classification' | 'Chat' | 'QA' | 'Extraction';
  output_type: 'JSON' | 'text' | 'label' | 'multi-label';
  domain: 'General' | 'Finance' | 'Medical' | 'Legal' | 'Code';
  language: string;
  training_mode: 'fast' | 'balanced' | 'high_quality';
  training_method: 'lora' | 'qlora' | 'full';
  base_model: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  max_seq_len: number;
  lora_rank: 8 | 16 | 32 | 64;
  gradient_accumulation: number;
  precision: 'bf16' | 'fp16' | 'float32';
  early_stopping: boolean;
  class_balancing: boolean;
  data_augmentation: boolean;
  resume_checkpoint?: string | null;
};

export type Pipeline = {
  id: string;
  name: string;
  description?: string | null;
  node_graph: NodeGraph;
  config: PipelineConfig;
  is_agent_configured: boolean;
  reasoning: Record<string, string>;
  created_at: string;
  updated_at: string;
};

// ── Jobs ──────────────────────────────────────────────────────────────────

export type Job = {
  id: string;
  pipeline_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  current_loss: number | null;
  progress_pct: number;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
};

// ── Agent chat (legacy free-form) ─────────────────────────────────────────

export type ChatMessage = { role: 'user' | 'assistant'; content: string };

// ── Agent runtime: sessions + events ──────────────────────────────────────

export type FSMState =
  | 'init'
  | 'profiling'
  | 'clarifying'
  | 'planning'
  | 'awaiting_approval'
  | 'executing'
  | 'monitoring'
  | 'recovering'
  | 'evaluating'
  | 'awaiting_export_choice'
  | 'finalizing'
  | 'done'
  | 'failed'
  | 'cancelled';

export type ClarificationQuestion = {
  question_id: string;
  question: string;
  kind: 'single_choice' | 'multi_choice' | 'text' | 'yes_no';
  options: string[];
  context?: string | null;
  required: boolean;
};

export type ClarificationAnswer = {
  question_id: string;
  value: unknown;
  answered_at: string;
};

export type RecoveryRecord = {
  diff_id: string;
  operations: { op: string; path: string; old: unknown; new: unknown }[];
  rationale: string;
  confidence: number;
  estimated_extra_minutes: number;
  approved: boolean | null;
  created_at: string;
};

export type AgentSession = {
  id: string;
  name?: string;
  dataset_id: string;
  pipeline_id: string | null;
  job_id: string | null;
  state: FSMState;
  state_entered_at: string;
  confidence: number;
  question_budget_max: number;
  question_budget_used: number;
  pending_question: ClarificationQuestion | null;
  clarifications: ClarificationAnswer[];
  recovery_history: RecoveryRecord[];
  artifacts: Record<string, unknown>;
  llm_provider?: string | null;
  llm_model?: string | null;
  llm_base_url?: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type SessionListItem = {
  id: string;
  name?: string;
  dataset_id: string;
  state: FSMState;
  pipeline_id: string | null;
  job_id: string | null;
  confidence: number;
  created_at: string;
  updated_at: string;
};

export type EventKind =
  | 'SessionStarted' | 'SessionPaused' | 'SessionResumed' | 'SessionClosed'
  | 'DatasetUploaded' | 'IntakeStarted' | 'IntakeCompleted'
  | 'DatasetProfileStarted' | 'DatasetProfileCompleted'
  | 'DataHealthReport'
  | 'TaskInferenceStarted' | 'TaskInferred' | 'IntentConfidenceLow'
  | 'UserClarificationRequested' | 'UserClarificationReceived'
  | 'HardwareProfileStarted' | 'HardwareProfileCompleted'
  | 'CandidateModelsRanked' | 'StrategyChosen'
  | 'PipelineDraftRequested' | 'PipelineDraftCreated'
  | 'PipelineApprovalRequested' | 'PipelineApproved' | 'PipelineRejected'
  | 'PipelineExecutionStarted'
  | 'TrainingMetricUpdated'
  | 'TrainingAnomalyDetected' | 'RecoveryPlanGenerated'
  | 'RetryRequested' | 'RetryApproved' | 'RetryDenied'
  | 'TrainingCompleted' | 'EvaluationStarted' | 'EvaluationCompleted'
  | 'ExportChoiceRequested'
  | 'SaveLocalRequested' | 'PushToHFRequested' | 'ExportCompleted'
  | 'PhaseStarted' | 'PhaseCompleted'
  | 'PhasePlanProposed' | 'PhaseApproved' | 'PhaseCommented'
  | 'NodeMaterialized' | 'EdgeMaterialized'
  | 'DataHealthReport'
  | 'DatasetRestructured'
  | 'AgentThinking' | 'AgentPlanning' | 'AgentAsking' | 'AgentGarnishing' | 'AgentExecuting'
  | 'BlackboardUpdated'
  | 'AuditCritique' | 'AuditOverride'
  | 'SandboxBenchmarkStarted' | 'SandboxBenchmarkCompleted'
  | 'CircuitBreakerTripped' | 'CheckpointSaved'
  | 'AssistantMessage' | 'UserMessage'
  | 'AgentToolCalled' | 'AgentDecisionRecorded' | 'Error';

export type StreamTone = 'thinking' | 'planning' | 'asking' | 'garnishing' | 'executing';

export type AgentEvent = {
  id: string;
  session_id: string;
  kind: EventKind;
  actor: string;
  parent_event_id?: string | null;
  payload: Record<string, unknown>;
  rationale?: string | null;
  confidence?: number | null;
  decision_id?: string | null;
  created_at: string;
};

// ── Models ────────────────────────────────────────────────────────────────

export type ModelRecord = {
  id: string;
  kind: 'base' | 'trained';
  name?: string;
  repo_id?: string;
  job_id?: string;
  local_path?: string;
  hf_repo_id?: string;
  is_pushed_to_hub?: boolean;
  created_at: string;
};
