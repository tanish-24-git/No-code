# FineTune Studio — Frontend Prompt for Claude Code

> Paste this into Claude Code to wire the frontend HTML to the live backend API.

---

## CONTEXT

The static HTML template (`index.html`) is already built. Your job is to:
1. Wire all UI interactions to the FastAPI backend (`http://localhost:8000/api`)
2. Implement the real node drag-and-drop canvas with connection drawing
3. Implement real-time SSE log streaming from training jobs
4. Make the Settings page save/load from the backend
5. Implement dataset upload with progress
6. Implement AI agent auto-config that fills node fields from API response

The frontend stays as a single `index.html` — no build tools, no bundler.
Use vanilla JS + fetch API only. No React, no Vue, no npm.

---

## API BASE URL

```javascript
const API = 'http://localhost:8000/api';
```

---

## FEATURE 1: Settings Page — API Key Management

**On page load:**
```javascript
// GET /api/settings → populate all fields, show masked keys
fetch(`${API}/settings`)
  .then(r => r.json())
  .then(data => {
    // Show masked agent key (e.g. "****ab1c")
    // Show masked HF token
    // Set all toggle states
    // Set selected model radio
  });
```

**Verify Agent Key button:**
```javascript
// POST /api/settings/verify-agent { api_key, provider }
// On success: show green "● verified" status
// On failure: show red "● invalid key"
```

**Verify HF Token button:**
```javascript
// POST /api/settings/verify-hf { token }
// On success: show "● verified · username" in green
// Update nav status dot to show HF username
```

**Save Settings button:**
```javascript
// PUT /api/settings with all form values
// Show brief "Saved" toast notification
```

---

## FEATURE 2: Dataset Upload

**Upload zone click/drop handler:**
```javascript
// POST /api/datasets/upload (multipart/form-data, field: "file")
// Show upload progress bar during upload
// On success: 
//   - Show filename + row count in Dataset node
//   - If agent is configured: show "Agent analyzing..." spinner in toolbar
//   - Poll GET /api/datasets/{id} until is_analyzed=true
//   - Then call applyAgentConfig(analysisResult)
```

**`applyAgentConfig(config)`:**
```javascript
// Receives the 22-field config from the agent
// Animates filling in each node field one by one (200ms delay between each)
// Show green flash on each field as it's filled
// Log each change to the log panel with reasoning if show_agent_reasoning=true
// After all fields filled: show "Agent configured pipeline" success message
```

---

## FEATURE 3: Node Canvas — Drag & Drop

Implement a proper interactive canvas:

```javascript
// Node dragging
// - mousedown on node header: start drag
// - mousemove: update node position (transform: translate)
// - mouseup: save final position
// - Store all node positions in pipelineState object

// Connection drawing  
// - mousedown on output port (blue dot): start drawing connection
// - Draw SVG bezier curve following mouse
// - mouseup on input port: snap connection, save to pipelineState.connections
// - Render all connections as SVG cubic bezier paths
// - Connections update live as nodes are dragged

// Node selection
// - click node: highlight with green border, show in Inspector panel
// - Inspector panel shows that node's field values
// - Editing in Inspector syncs back to node fields in real-time

// Canvas pan
// - mousedown on empty canvas: start panning
// - mousemove: translate canvas-inner
// - mousewheel: zoom (scale transform on canvas-inner, min 0.4 max 2.0)

// State structure:
const pipelineState = {
  nodes: [
    { id: 'node1', type: 'dataset', x: 48, y: 80, fields: {} },
    // ...
  ],
  connections: [
    { from: 'node1', fromPort: 'out', to: 'node2', toPort: 'in' },
    // ...
  ],
  selectedNodeId: null
};
```

---

## FEATURE 4: Pipeline Save & Load

```javascript
// Auto-save pipeline to backend every 10 seconds if changed
// PUT /api/pipelines/{id} with { node_graph: pipelineState, config: flatConfig }

// On page load: GET /api/pipelines → load most recent pipeline
// Restore node positions and field values from saved state
```

---

## FEATURE 5: Run Pipeline + SSE Log Streaming

**Run button click:**
```javascript
async function runPipeline() {
  // 1. Validate: check dataset is uploaded, base_model is set
  // 2. POST /api/jobs/start { pipeline_id }
  // 3. Get job_id from response
  // 4. Open SSE connection: new EventSource(`${API}/jobs/${jobId}/logs`)
  // 5. Append each SSE message to log panel
  // 6. Poll GET /api/jobs/${jobId} every 2s for metrics (loss, epoch, progress)
  // 7. Update Inspector panel with live metrics
  // 8. On [DONE]: close SSE, show completion UI
}

// SSE handler:
const evtSource = new EventSource(`${API}/jobs/${jobId}/logs`);
evtSource.onmessage = (e) => {
  if (e.data === '[DONE]') {
    evtSource.close();
    showCompletionBanner();
    return;
  }
  appendToLogPanel(e.data, inferLogType(e.data));
};
```

**`inferLogType(msg)`:** returns 'success'|'warn'|'error'|'info' based on keywords

---

## FEATURE 6: Model Export & HF Hub Push

After job completes, show a completion banner:
```html
<!-- Completion banner (append to canvas area) -->
<div class="completion-banner">
  <span>✓ Training complete</span>
  <button onclick="exportModel()">↓ Save Locally</button>
  <button onclick="pushToHub()">⤴ Push to HF Hub</button>
</div>
```

```javascript
async function exportModel() {
  // POST /api/models/{id}/export
  // Show "Saved to ./models/{job_id}/" in log panel
}

async function pushToHub() {
  // Check HF token is verified first (GET /api/settings)
  // If not: show "Set HF token in Settings first"
  // POST /api/models/{id}/push-hub { repo_id: inspectorRepoId }
  // Poll GET /api/models/{id}/push-status
  // Show progress in log panel
}
```

---

## FEATURE 7: Health Check & Status Bar

On page load and every 30 seconds:
```javascript
// GET /health
// Update nav status bar:
// - If GPU: "local · cuda · {gpu_name}"
// - If CPU: "local · cpu"
// - If backend unreachable: show red dot + "backend offline"
```

---

## UI COMPONENTS TO ADD/ENHANCE

### Toast notification:
```javascript
function showToast(msg, type='info') {
  // Small pill that slides in from bottom-right
  // Auto-dismisses after 3 seconds
  // Types: info (grey), success (green), error (red)
}
```

### Upload progress bar:
```javascript
// Use XMLHttpRequest (not fetch) for upload progress events
// Show a thin progress bar at the top of the Dataset node
// Animate from 0 to 100% during upload
```

### Agent config animation:
```javascript
// When agent config arrives, flash each node's color-bar briefly
// Fill each field with a typewriter effect (character by character)
// Add a subtle green glow to the whole node that fades out
```

### Node mini-status in canvas:
```javascript
// When job is running:
// - Highlight the active node with amber border + "running" label
// - Pulse the connector ports of active node
// - Mark completed nodes with green border + "✓"
```

---

## ERROR HANDLING

```javascript
// All API calls wrapped in try/catch
// Network errors → showToast("Backend offline", "error")
// 400/422 errors → showToast(error.message, "error")
// 500 errors → showToast("Server error — check Docker logs", "error")
// Log all errors to log panel as red lines
```

---

## STATE PERSISTENCE (localStorage)

Store locally for instant page load:
```javascript
localStorage.setItem('lastPipelineId', id);
localStorage.setItem('canvasZoom', zoom);
localStorage.setItem('canvasOffset', JSON.stringify({x,y}));
// Restore on page load before API fetch
```

---

## FINAL CHECKLIST

- [ ] Settings page saves and loads from backend
- [ ] Agent key verify shows green/red status
- [ ] HF token verify shows username
- [ ] Dataset drag-and-drop upload works with progress
- [ ] Agent auto-fills all node fields with animation
- [ ] Nodes are draggable with live connection lines
- [ ] Canvas pans and zooms smoothly
- [ ] Run button starts job and streams logs via SSE
- [ ] Live metrics (loss, epoch) update in Inspector
- [ ] Completion banner with Export and Push buttons
- [ ] Toast notifications for all actions
- [ ] Health check updates status bar
- [ ] Works fully offline (no CDN deps except Google Fonts)
