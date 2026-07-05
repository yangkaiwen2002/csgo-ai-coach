import Link from "next/link";
import Navbar from "@/components/Navbar";

const STEPS = [
  {
    num: "01",
    title: "Parse Replay",
    desc: "awpy extracts player positions, kills, and events from .dem files at 8 samples/sec",
    icon: (
      <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
    ),
  },
  {
    num: "02",
    title: "Extract Features",
    desc: "23-dim limited-information vector: only what the player can actually see or hear",
    icon: (
      <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9h16.5m-16.5 6.75h16.5" />
      </svg>
    ),
  },
  {
    num: "03",
    title: "Win-Rate Labeling",
    desc: "No human annotation: pro match outcomes label each state. Highest win-rate action = ground truth",
    icon: (
      <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
      </svg>
    ),
  },
  {
    num: "04",
    title: "Transformer Inference",
    desc: "Encoder-only Transformer over a 32-tick window outputs one of 9 tactical action classes",
    icon: (
      <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
      </svg>
    ),
  },
];

const TECH = [
  { name: "PyTorch", color: "text-orange-300 bg-orange-300/8 border-orange-300/20" },
  { name: "Transformer", color: "text-blue-300 bg-blue-300/8 border-blue-300/20" },
  { name: "FastAPI", color: "text-emerald-300 bg-emerald-300/8 border-emerald-300/20" },
  { name: "awpy", color: "text-purple-300 bg-purple-300/8 border-purple-300/20" },
  { name: "Next.js", color: "text-gray-300 bg-gray-300/8 border-gray-300/20" },
  { name: "NumPy", color: "text-sky-300 bg-sky-300/8 border-sky-300/20" },
];

const ACTIONS_PREVIEW = [
  { action: "ROTATE B", color: "text-orange-400 bg-orange-400/10 border-orange-400/25" },
  { action: "HOLD A", color: "text-blue-400 bg-blue-400/10 border-blue-400/25" },
  { action: "PUSH A", color: "text-red-400 bg-red-400/10 border-red-400/25" },
  { action: "SMOKE B", color: "text-purple-400 bg-purple-400/10 border-purple-400/25" },
  { action: "FALL BACK", color: "text-gray-400 bg-gray-400/10 border-gray-400/25" },
];

const FEATURES = [
  "Team movement direction",
  "Gunfire heard (by area)",
  "Teammate rotation signals",
  "Economic state",
  "Time pressure",
  "Player alive counts",
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#06060a] text-gray-100">
      <Navbar />

      {/* Hero */}
      <section className="relative pt-32 pb-24 px-6 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-orange-500/5 rounded-full blur-3xl pointer-events-none" />
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.15]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.04) 1px,transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="relative max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-orange-400/20 bg-orange-400/8 text-orange-400 text-xs font-mono mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
            Self-supervised · No human annotation
          </div>

          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight tracking-tight">
            AI that reviews your{" "}
            <span className="text-orange-400">CS:GO replays</span>
          </h1>

          <p className="text-lg text-gray-400 mb-4 max-w-2xl mx-auto leading-relaxed">
            Upload a .dem file. Get per-round coaching:{" "}
            <span className="text-gray-200 italic">
              &ldquo;At tick 192 with 71s left, you should have rotated B — 87% confidence.&rdquo;
            </span>
          </p>

          <p className="text-sm text-gray-600 mb-10">
            Trained on pro matches from HLTV · Labels derived from win rates, not human annotation
          </p>

          <div className="flex flex-wrap justify-center gap-2 mb-10">
            {ACTIONS_PREVIEW.map(({ action, color }) => (
              <span
                key={action}
                className={`text-xs font-mono font-semibold tracking-wider px-3 py-1.5 rounded border ${color}`}
              >
                {action}
              </span>
            ))}
            <span className="text-xs text-gray-600 self-center">+ 4 more</span>
          </div>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link
              href="/demo"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-orange-500 hover:bg-orange-400 text-white font-semibold text-sm transition-colors duration-150"
            >
              View Interactive Demo
              <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </Link>
            <a
              href="https://github.com/yangkaiwen2002/csgo-ai-coach"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-white/12 hover:border-white/20 text-gray-300 hover:text-white font-semibold text-sm transition-colors duration-150"
            >
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
              View Source
            </a>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-20 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <p className="text-xs text-gray-600 uppercase tracking-widest text-center mb-3">Architecture</p>
          <h2 className="text-2xl font-bold text-center text-white mb-12">How it works</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {STEPS.map((step, i) => (
              <div
                key={i}
                className="rounded-xl border border-white/8 bg-white/2 p-6 hover:border-white/14 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-lg bg-orange-400/10 border border-orange-400/20 flex items-center justify-center text-orange-400 flex-shrink-0">
                    {step.icon}
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-mono text-gray-600">{step.num}</span>
                      <span className="text-sm font-semibold text-white">{step.title}</span>
                    </div>
                    <p className="text-sm text-gray-500 leading-relaxed">{step.desc}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Key insight */}
      <section className="py-20 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <div className="rounded-2xl border border-orange-400/15 bg-orange-400/4 p-8 md:p-10">
            <p className="text-xs text-orange-400 uppercase tracking-widest mb-4 font-mono">
              Core Design Decision
            </p>
            <h3 className="text-xl font-bold text-white mb-4">
              Limited-information inputs only
            </h3>
            <p className="text-gray-400 leading-relaxed mb-6">
              The model only sees what a real player can perceive: gunfire direction, teammate positions,
              footsteps heard, and player counts. No &ldquo;God view.&rdquo; This makes the coaching
              advice realistic and applicable — the model learned from the same partial information
              that pros play with.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {FEATURES.map((f) => (
                <div key={f} className="flex items-center gap-2 text-sm text-gray-400">
                  <span className="w-1 h-1 rounded-full bg-orange-400 flex-shrink-0" />
                  {f}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Tech stack */}
      <section className="py-20 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-xs text-gray-600 uppercase tracking-widest mb-8">Built with</p>
          <div className="flex flex-wrap justify-center gap-3">
            {TECH.map(({ name, color }) => (
              <span key={name} className={`px-4 py-2 rounded-lg border text-sm font-medium ${color}`}>
                {name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 border-t border-white/5">
        <div className="max-w-xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">See it in action</h2>
          <p className="text-gray-500 mb-8">
            The interactive demo uses synthetic match data to show exactly what coaching output
            looks like — no .dem file needed.
          </p>
          <Link
            href="/demo"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-lg bg-orange-500 hover:bg-orange-400 text-white font-semibold transition-colors duration-150"
          >
            Open Demo Dashboard
            <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-6">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <p className="text-xs text-gray-700">CS:GO AI Coach — ML Systems Project</p>
          <a
            href="https://github.com/yangkaiwen2002/csgo-ai-coach"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            github.com/yangkaiwen2002/csgo-ai-coach
          </a>
        </div>
      </footer>
    </div>
  );
}
