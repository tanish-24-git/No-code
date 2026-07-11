/**
 * Dev-only OpenAI-compatible mock endpoint for harness testing without a real
 * key. Enabled ONLY when FT_ENABLE_MOCK_LLM=1. Point the harness at it with:
 *   LLM_BASE_URL=http://localhost:3000/api/dev/mock-llm/v1
 * Streams a canned completion (with usage on the final chunk), echoing the
 * last user message so tests can assert round-tripping.
 */
export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  if (process.env.FT_ENABLE_MOCK_LLM !== '1') {
    return Response.json({ error: 'mock disabled' }, { status: 404 });
  }
  const body = (await req.json().catch(() => ({}))) as {
    messages?: { role: string; content: unknown }[];
    stream?: boolean;
    stream_options?: unknown;
    reasoning_effort?: string;
  };
  const lastUser = [...(body.messages ?? [])].reverse().find((m) => m.role === 'user');
  const userText =
    typeof lastUser?.content === 'string'
      ? lastUser.content
      : ((lastUser?.content as { text?: string }[] | undefined) ?? []).map((p) => p.text ?? '').join(' ');
  const reply = `MOCK-REPLY to: "${String(userText).slice(0, 80)}" (stream_options=${JSON.stringify(
    body.stream_options ?? null,
  )}, reasoning_effort=${body.reasoning_effort ?? 'none'})`;
  const words = reply.split(' ');

  if (!body.stream) {
    return Response.json({
      id: 'mock-1',
      object: 'chat.completion',
      model: 'mock-model',
      choices: [{ index: 0, message: { role: 'assistant', content: reply }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 123, completion_tokens: words.length, total_tokens: 123 + words.length },
    });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
      for (let i = 0; i < words.length; i++) {
        send({
          id: 'mock-1',
          object: 'chat.completion.chunk',
          model: 'mock-model',
          choices: [{ index: 0, delta: { content: (i > 0 ? ' ' : '') + words[i] }, finish_reason: null }],
        });
        await new Promise((r) => setTimeout(r, 25));
      }
      send({
        id: 'mock-1',
        object: 'chat.completion.chunk',
        model: 'mock-model',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      });
      send({
        id: 'mock-1',
        object: 'chat.completion.chunk',
        model: 'mock-model',
        choices: [],
        usage: { prompt_tokens: 123, completion_tokens: words.length, total_tokens: 123 + words.length },
      });
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
