import Link from 'next/link';

export default function HomePage() {
  return (
    <section className="relative flex flex-col items-center justify-center min-h-[calc(100vh-52px)] text-center px-6">
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          background:
            'radial-gradient(ellipse 500px 500px at center, rgba(74,222,128,0.05), transparent 70%)',
        }}
      />
      <span className="pill mb-10">
        <span className="dot dot-success" />
        v2 · open source · bring your own inference
      </span>
      <h1 className="font-sans font-extrabold leading-[1.05] tracking-[-2px] text-[clamp(42px,7vw,72px)] mb-5">
        Fine-tune anything.
        <br />
        <span className="text-fg-2">Talk to your inference.</span>
      </h1>
      <p className="max-w-[560px] text-fg-2 text-sm leading-relaxed mb-10">
        Drop a dataset. Sketch a node-graph pipeline. Have a real conversation with an
        agent that can read your local inference endpoints and tell you exactly which
        metrics to set. Everything runs on your machine — no Redis, no DB, no Docker
        required.
      </p>
      <div className="flex items-center gap-3">
        <Link href="/playground" className="btn btn-primary">
          open playground →
        </Link>
        <Link href="/settings" className="btn">
          set up keys
        </Link>
      </div>

      <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-[920px] w-full text-left">
        <FeatureCard
          title="node-based pipelines"
          body="Drag-and-drop dataset → preprocess → train → evaluate → export. Wire it however you want."
        />
        <FeatureCard
          title="byo inference"
          body="Register Ollama, OpenAI-compat, HF Inference, or Anthropic endpoints. The agent reads them as tools."
        />
        <FeatureCard
          title="agent that listens"
          body="Chat about your hardware, dataset, and endpoints. The agent suggests metrics and writes them back."
        />
      </div>
    </section>
  );
}

function FeatureCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="card">
      <div className="text-[11px] uppercase tracking-wider text-fg-2 mb-2">{title}</div>
      <p className="text-xs leading-relaxed text-fg">{body}</p>
    </div>
  );
}
