/**
 * Hooks for consuming agent-runtime SSE streams. The session events endpoint
 * sends one frame per AgentEvent: `event: <Kind>\ndata: <json>\n\n`. We parse
 * those into typed AgentEvent objects and append them to a list.
 */
'use client';

import { useEffect, useRef, useState } from 'react';
import { sseUrl } from './api';
import type { AgentEvent } from './types';

export function useSessionEvents(sessionId: string | null | undefined) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [closed, setClosed] = useState(false);
  const sessionRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    // Reset state when a new session is bound.
    sessionRef.current = sessionId;
    setEvents([]);
    setConnected(false);
    setClosed(false);

    const es = new EventSource(sseUrl(`/api/sessions/${sessionId}/events`));
    const handle = (kind: string) => (e: MessageEvent) => {
      if (sessionRef.current !== sessionId) return;
      try {
        const payload = JSON.parse(e.data) as AgentEvent;
        setEvents((prev) => [...prev, payload]);
        if (kind === 'SessionClosed') setClosed(true);
      } catch {
        // Ignore malformed frames; the keepalive comments are filtered out.
      }
    };

    // EventSource only fires `message` for unnamed events; named events need
    // explicit listeners. The backend always emits a name, so we listen to
    // every kind we care about. Unknown kinds still come through `message`.
    const KNOWN: string[] = [
      'SessionStarted', 'SessionClosed',
      'DatasetUploaded', 'IntakeStarted', 'IntakeCompleted',
      'DatasetProfileStarted', 'DatasetProfileCompleted',
      'DataHealthReport',
      'TaskInferred', 'IntentConfidenceLow',
      'UserClarificationRequested', 'UserClarificationReceived',
      'HardwareProfileStarted', 'HardwareProfileCompleted',
      'CandidateModelsRanked', 'StrategyChosen',
      'PipelineDraftRequested', 'PipelineDraftCreated',
      'PipelineApprovalRequested', 'PipelineApproved', 'PipelineRejected',
      'PipelineExecutionStarted',
      'TrainingMetricUpdated', 'TrainingAnomalyDetected',
      'RecoveryPlanGenerated', 'RetryRequested', 'RetryApproved', 'RetryDenied',
      'TrainingCompleted', 'EvaluationStarted', 'EvaluationCompleted',
      'ExportChoiceRequested', 'SaveLocalRequested', 'PushToHFRequested',
      'ExportCompleted',
      'PhaseStarted', 'PhaseCompleted', 'PhasePlanProposed', 'PhaseApproved', 'PhaseCommented',
      'NodeMaterialized',
      'AgentThinking', 'AgentPlanning', 'AgentAsking', 'AgentGarnishing', 'AgentExecuting',
      'BlackboardUpdated',
      'AuditCritique', 'AuditOverride',
      'SandboxBenchmarkStarted', 'SandboxBenchmarkCompleted',
      'CircuitBreakerTripped', 'CheckpointSaved',
      'AssistantMessage', 'UserMessage',
      'AgentToolCalled', 'AgentDecisionRecorded', 'Error',
    ];
    KNOWN.forEach((k) => es.addEventListener(k, handle(k)));

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    return () => {
      es.close();
      sessionRef.current = null;
    };
  }, [sessionId]);

  return { events, connected, closed };
}
