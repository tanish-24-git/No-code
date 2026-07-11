/**
 * Dev-only OpenAI-compatible mock endpoint for harness testing without a real
 * key. Enabled ONLY when FT_ENABLE_MOCK_LLM=1. Point the harness at it with:
 *   LLM_BASE_URL=http://localhost:3000/api/dev/mock-llm/v1
 *
 * Behaviors (scripted, for E2E tests):
 *  - last message is a tool result        -> streams "TOOL-RESULT-ACK: <preview>"
 *  - user text is `use tool <name> <json>`-> streams a tool_calls turn for it
 *  - otherwise                            -> streams "MOCK-REPLY to: ..." echo
 * Always ends with a usage chunk (include_usage contract).
 */
export const dynamic = 'force-dynamic';

interface OaiMessage {
  role: string;
  content: unknown;
  tool_call_id?: string;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map((p) => (p as { text?: string }).text ?? '').join(' ');
  return '';
}

export async function POST(req: Request) {
  if (process.env.FT_ENABLE_MOCK_LLM !== '1') {
    return Response.json({ error: 'mock disabled' }, { status: 404 });
  }
  const body = (await req.json().catch(() => ({}))) as {
    messages?: OaiMessage[];
    stream?: boolean;
    stream_options?: unknown;
    reasoning_effort?: string;
  };
  const messages = body.messages ?? [];
  const last = messages[messages.length - 1];
  const lastUser = [...messages].reverse().find((m) => m.role === 'user');
  const userText = textOf(lastUser?.content).replace(/^\[user interjection\]\s*/, '');

  // Scripted tool calls: one `use tool <name> <json-args>` per line; multiple
  // lines in one message become multiple tool calls in ONE turn (parallel fan-out).
  const toolMatches =
    last?.role !== 'tool' ? [...userText.matchAll(/^use tool (\w+)\s+(\{.*\})\s*$/gm)] : [];

  const encoder = new TextEncoder();
  const chunks: unknown[] = [];
  const push = (delta: Record<string, unknown>, finish: string | null = null) =>
    chunks.push({
      id: 'mock-1',
      object: 'chat.completion.chunk',
      model: 'mock-model',
      choices: [{ index: 0, delta, finish_reason: finish }],
    });

  // Real providers guarantee syntactically valid JSON in function.arguments —
  // normalize through parse+stringify (tolerating raw newlines from test input).
  const calls: { name: string; args: string }[] = [];
  for (const m of toolMatches) {
    try {
      calls.push({ name: m[1], args: JSON.stringify(JSON.parse(m[2])) });
    } catch {
      // skip malformed scripted call
    }
  }

  if (calls.length > 0) {
    push({ role: 'assistant', content: `Calling ${calls.map((c) => c.name).join(' + ')} as instructed. ` });
    for (const [i, call] of calls.entries()) {
      push({
        tool_calls: [
          { index: i, id: `call_${Date.now()}_${i}`, type: 'function', function: { name: call.name, arguments: '' } },
        ],
      });
      // stream the arguments in two pieces like real providers do
      const mid = Math.floor(call.args.length / 2);
      push({ tool_calls: [{ index: i, function: { arguments: call.args.slice(0, mid) } }] });
      push({ tool_calls: [{ index: i, function: { arguments: call.args.slice(mid) } }] });
    }
    push({}, 'tool_calls');
  } else {
    const reply =
      last?.role === 'tool'
        ? `TOOL-RESULT-ACK: ${textOf(last.content).slice(0, 160).replace(/\s+/g, ' ')}`
        : `MOCK-REPLY to: "${userText.slice(0, 80)}" (stream_options=${JSON.stringify(
            body.stream_options ?? null,
          )}, reasoning_effort=${body.reasoning_effort ?? 'none'})`;
    for (const [i, word] of reply.split(' ').entries()) {
      push({ content: (i > 0 ? ' ' : '') + word });
    }
    push({}, 'stop');
  }
  chunks.push({
    id: 'mock-1',
    object: 'chat.completion.chunk',
    model: 'mock-model',
    choices: [],
    usage: { prompt_tokens: 123, completion_tokens: 42, total_tokens: 165 },
  });

  if (!body.stream) {
    // Non-streaming path (compaction summarize etc.) — text only.
    return Response.json({
      id: 'mock-1',
      object: 'chat.completion',
      model: 'mock-model',
      choices: [
        { index: 0, message: { role: 'assistant', content: 'MOCK-NONSTREAM-REPLY' }, finish_reason: 'stop' },
      ],
      usage: { prompt_tokens: 123, completion_tokens: 42, total_tokens: 165 },
    });
  }

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const c of chunks) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(c)}\n\n`));
        await new Promise((r) => setTimeout(r, 15));
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
    },
  });
}
