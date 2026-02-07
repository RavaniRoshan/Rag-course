import Link from "next/link";
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Layers,
  Sparkles,
  Terminal,
} from "lucide-react";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <section className="relative overflow-hidden border-b bg-background">
        <div className="container mx-auto grid gap-12 px-4 py-20 lg:grid-cols-[1.1fr_0.9fr] lg:py-28">
          <div className="flex flex-col justify-center space-y-6">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border bg-muted/60 px-4 py-1 text-xs font-medium text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              Next.js template for modern RAG teams
            </div>
            <div className="space-y-4">
              <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
                Build production-ready RAG systems faster.
              </h1>
              <p className="max-w-xl text-base text-muted-foreground sm:text-lg">
                Re-imagined for teams shipping with Next.js: learn chunking, retrieval, re-ranking,
                and agentic workflows through polished lessons, practical labs, and real-world
                templates.
              </p>
            </div>
            <div className="flex flex-wrap gap-4">
              <Link
                href="/introduction"
                className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Start the course
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
              <Link
                href="https://github.com"
                target="_blank"
                className="inline-flex h-12 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                View on GitHub
              </Link>
            </div>
            <div className="flex flex-wrap gap-6 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                10+ guided labs
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                4 production blueprints
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Ship-ready checklists
              </span>
            </div>
          </div>
          <div className="relative flex items-center justify-center">
            <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-primary/10 via-transparent to-primary/20 blur-3xl" />
            <div className="relative w-full rounded-3xl border bg-card p-6 shadow-xl">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="font-semibold text-primary">RAG Studio</span>
                <span>v2.1</span>
              </div>
              <div className="mt-6 space-y-4">
                <div className="rounded-xl border border-dashed bg-muted/50 p-4 text-sm text-muted-foreground">
                  Drag your data sources here
                </div>
                <div className="rounded-xl border bg-background p-4">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <span className="h-2 w-2 rounded-full bg-emerald-500" />
                    Retrieval pipeline
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Chunking + hybrid search + reranking
                  </p>
                </div>
                <div className="rounded-xl border bg-background p-4">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <span className="h-2 w-2 rounded-full bg-sky-500" />
                    Agentic QA loop
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Tools, memory, and evals wired in
                  </p>
                </div>
              </div>
              <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
                <span>Latency: 320ms</span>
                <span>Grounding: 98%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-b bg-muted/30">
        <div className="container mx-auto flex flex-col gap-6 px-4 py-10 text-center md:flex-row md:items-center md:justify-between">
          <p className="text-sm font-medium uppercase tracking-[0.3em] text-muted-foreground">
            Trusted by builders shipping RAG in production
          </p>
          <div className="flex flex-wrap items-center justify-center gap-6 text-sm font-semibold text-muted-foreground">
            <span>Atlas AI</span>
            <span>Vector Works</span>
            <span>PromptCloud</span>
            <span>Graph Labs</span>
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16 md:py-24">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
            What you&apos;ll master
          </p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            A complete template for modern RAG teams
          </h2>
          <p className="mt-4 text-base text-muted-foreground">
            Every lesson pairs strategy with code. Build the pipelines, evaluate them, and ship the
            same patterns as top AI product teams.
          </p>
        </div>
        <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[
            {
              title: "Deep dive modules",
              description:
                "From query planning to GraphRAG, each module is tuned for production impact.",
              icon: Layers,
            },
            {
              title: "Code-first labs",
              description:
                "Hands-on notebooks and Next.js-ready examples for every concept you learn.",
              icon: Terminal,
            },
            {
              title: "Visual learning",
              description:
                "Architecture diagrams and pipeline maps that clarify complex systems fast.",
              icon: BookOpen,
            },
            {
              title: "Evaluation playbooks",
              description:
                "Grounding metrics, retrieval scorecards, and QA harnesses to prove quality.",
              icon: CheckCircle2,
            },
            {
              title: "Deployment blueprints",
              description:
                "Secure, scalable reference stacks for launching RAG apps in production.",
              icon: Sparkles,
            },
            {
              title: "Team-ready templates",
              description:
                "Reusable prompts, toolchains, and checklists to align your team quickly.",
              icon: ArrowRight,
            },
          ].map((feature) => (
            <div
              key={feature.title}
              className="flex h-full flex-col gap-4 rounded-2xl border bg-card p-6 text-left shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <feature.icon className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">{feature.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t bg-primary/5">
        <div className="container mx-auto grid gap-8 px-4 py-16 md:grid-cols-[1.2fr_0.8fr] md:items-center">
          <div>
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Ready to launch your next RAG experience?
            </h2>
            <p className="mt-3 text-base text-muted-foreground">
              Follow a step-by-step path from fundamentals to production. Get the same playbooks
              we use to ship retrieval systems that scale.
            </p>
          </div>
          <div className="flex flex-wrap gap-4 md:justify-end">
            <Link
              href="/introduction"
              className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Jump in now
            </Link>
            <Link
              href="https://github.com"
              target="_blank"
              className="inline-flex h-12 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Browse the repo
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
