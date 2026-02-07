import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Layers,
  Sparkles,
  Smartphone,
  Users,
} from "lucide-react";

const features = [
  {
    title: "Opportunities",
    description:
      "Surface market moves, trendlines, and breakout alerts tailored to your watchlist.",
    icon: Sparkles,
  },
  {
    title: "Execution",
    description:
      "Deploy trades in seconds with guided workflows, smart sizing, and safety rails.",
    icon: CheckCircle2,
  },
  {
    title: "Lifted Options",
    description:
      "Access curated options plays with guardrails, clarity, and real-time scenarios.",
    icon: Layers,
  },
];

const testimonials = [
  {
    quote:
      "Slush turned my scattered watchlist into a plan. The alerts feel like having a trader in my pocket.",
    name: "Taylor R.",
    title: "Growth marketer",
  },
  {
    quote:
      "I finally understand what I'm doing. The onboarding and prompts kept me confident.",
    name: "Priya L.",
    title: "Product designer",
  },
  {
    quote:
      "Execution is fast, and the risk checks are baked in. I don't feel lost anymore.",
    name: "Marcus J.",
    title: "Indie founder",
  },
];

const audiences = [
  {
    title: "First-timers",
    description: "Get step-by-step explainers and bite-size wins without the jargon.",
  },
  {
    title: "Portfolio builders",
    description: "Track positions, rebalance quickly, and stay ahead of momentum shifts.",
  },
  {
    title: "Signal hunters",
    description: "Lean on curated flows that turn complex data into clear decisions.",
  },
];

const ecosystems = [
  "Solana",
  "Arbitrum",
  "Base",
  "Optimism",
  "Ethereum",
  "Polygon",
  "Aptos",
  "Sui",
];

export default function LandingPage() {
  return (
    <div className="flex flex-col">
      <section className="relative overflow-hidden border-b bg-background">
        <div className="container mx-auto grid gap-12 px-4 py-20 lg:grid-cols-[1.1fr_0.9fr] lg:py-28">
          <div className="flex flex-col justify-center space-y-6">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border bg-muted/60 px-4 py-1 text-xs font-medium text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              SLUSH by creators, for humans
            </div>
            <div className="space-y-4">
              <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
                Trade crypto with clarity, not chaos.
              </h1>
              <p className="max-w-xl text-base text-muted-foreground sm:text-lg">
                Slush turns market noise into guided moves, smart execution, and explainable
                opportunities so you can act with confidence.
              </p>
            </div>
            <div className="flex flex-wrap gap-4">
              <Link
                href="/introduction"
                className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Get Slush
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
              <Link
                href="https://github.com"
                target="_blank"
                className="inline-flex h-12 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Join the waitlist
              </Link>
            </div>
            <div className="flex flex-wrap gap-6 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Live market guidance
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Built-in guardrails
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Human-friendly crypto
              </span>
            </div>
          </div>
          <div className="relative flex items-center justify-center">
            <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-primary/10 via-transparent to-primary/20 blur-3xl" />
            <div className="relative w-full rounded-3xl border bg-card p-6 shadow-xl">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="font-semibold text-primary">SLUSH Terminal</span>
                <span>v1.0</span>
              </div>
              <div className="mt-6 space-y-4">
                <div className="rounded-xl border border-dashed bg-muted/50 p-4 text-sm text-muted-foreground">
                  Drop your watchlist to unlock insights
                </div>
                <div className="rounded-xl border bg-background p-4">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <span className="h-2 w-2 rounded-full bg-emerald-500" />
                    Opportunity pulse
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Momentum signals with explainers
                  </p>
                </div>
                <div className="rounded-xl border bg-background p-4">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <span className="h-2 w-2 rounded-full bg-sky-500" />
                    Execution planner
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Smart order sizing + stop guides
                  </p>
                </div>
              </div>
              <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
                <span>Win rate: 71%</span>
                <span>Safety score: 9.2</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16 md:py-24">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
            The edge
          </p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            Move faster with Slush intelligence.
          </h2>
          <p className="mt-4 text-base text-muted-foreground">
            Everything you need to spot, plan, and execute crypto trades lives in one guided hub.
          </p>
        </div>
        <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="flex h-full flex-col gap-4 rounded-2xl border bg-card p-6 text-left shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <feature.icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">
                  {feature.title}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t bg-primary/5">
        <div className="container mx-auto grid gap-8 px-4 py-16 md:grid-cols-[1.2fr_0.8fr] md:items-center">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
              Onboarding
            </p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
              Get trading-ready in minutes.
            </h2>
            <p className="mt-3 text-base text-muted-foreground">
              Answer a few questions, connect your favorite networks, and let Slush build your
              personalized playbook.
            </p>
            <div className="mt-6 flex flex-wrap gap-3 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Goals + risk profile
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Portfolio sync
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Actionable next steps
              </span>
            </div>
          </div>
          <div className="rounded-2xl border bg-background p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">
              Setup checklist
            </p>
            <ul className="mt-4 space-y-3 text-sm">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Choose your strategy style
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Connect wallets + exchanges
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Enable alert channels
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Review your first plan
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16 md:py-24">
        <div className="grid gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <div className="rounded-3xl border bg-card p-8 shadow-sm">
            <div className="flex items-center gap-3 text-primary">
              <Smartphone className="h-6 w-6" />
              <p className="text-sm font-semibold uppercase tracking-[0.3em]">Mobile app</p>
            </div>
            <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl">
              Your entire portfolio in your pocket.
            </h2>
            <p className="mt-3 text-base text-muted-foreground">
              Track positions, get instant opportunity alerts, and execute trades from the Slush
              mobile companion.
            </p>
            <div className="mt-6 flex flex-wrap gap-4">
              <Link
                href="/introduction"
                className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Download the app
              </Link>
              <Link
                href="https://github.com"
                target="_blank"
                className="inline-flex h-12 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                View demo
              </Link>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {[
              "Push notifications that explain why the signal matters.",
              "One-tap execution with built-in risk checks.",
              "Live sentiment and on-chain heatmaps.",
              "Personalized watchlists synced across devices.",
            ].map((item) => (
              <div key={item} className="rounded-2xl border bg-muted/30 p-5 text-sm">
                {item}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-y bg-muted/20">
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
              Don&apos;t believe us?
            </p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
              Traders who switched to Slush tell the story.
            </h2>
          </div>
          <div className="mt-12 grid gap-6 md:grid-cols-3">
            {testimonials.map((testimonial) => (
              <div
                key={testimonial.name}
                className="flex h-full flex-col justify-between rounded-2xl border bg-background p-6 shadow-sm"
              >
                <p className="text-sm text-muted-foreground">“{testimonial.quote}”</p>
                <div className="mt-6">
                  <p className="text-sm font-semibold text-foreground">{testimonial.name}</p>
                  <p className="text-xs text-muted-foreground">{testimonial.title}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16 md:py-24">
        <div className="mx-auto max-w-3xl text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
            Crypto for humans
          </p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            Built for every kind of crypto explorer.
          </h2>
          <p className="mt-4 text-base text-muted-foreground">
            Whether you&apos;re just getting started or running a diversified book, Slush adapts to
            your pace.
          </p>
        </div>
        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {audiences.map((audience) => (
            <div key={audience.title} className="rounded-2xl border bg-card p-6 shadow-sm">
              <div className="flex items-center gap-3 text-primary">
                <Users className="h-5 w-5" />
                <h3 className="text-lg font-semibold">{audience.title}</h3>
              </div>
              <p className="mt-3 text-sm text-muted-foreground">{audience.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t bg-primary/5">
        <div className="container mx-auto px-4 py-12 md:py-16">
          <div className="flex flex-col gap-6 text-center">
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
              Ecosystem
            </p>
            <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
              Connected across the chains you care about.
            </h2>
            <div className="mt-6 grid gap-4 text-sm font-semibold text-muted-foreground sm:grid-cols-2 md:grid-cols-4">
              {ecosystems.map((ecosystem) => (
                <div
                  key={ecosystem}
                  className="rounded-xl border bg-background px-4 py-3 shadow-sm"
                >
                  {ecosystem}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="container mx-auto px-4 py-16 md:py-24">
        <div className="rounded-3xl border bg-card p-10 text-center shadow-sm md:p-14">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
            Your inbox just got better
          </p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl">
            Weekly signals, market recaps, and Slush updates.
          </h2>
          <p className="mt-3 text-base text-muted-foreground">
            Join thousands of builders and traders who get the Slush briefing every Friday.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-4">
            <Link
              href="/introduction"
              className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Subscribe now
            </Link>
            <Link
              href="https://github.com"
              target="_blank"
              className="inline-flex h-12 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              View past issues
            </Link>
          </div>
        </div>
      </section>

      <section className="border-t bg-muted/30">
        <div className="container mx-auto flex flex-col items-center justify-between gap-6 px-4 py-12 text-center md:flex-row md:text-left">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-muted-foreground">
              Get Slush
            </p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">
              Start trading with confidence today.
            </h2>
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              href="/introduction"
              className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Get Slush
            </Link>
            <Link
              href="https://github.com"
              target="_blank"
              className="inline-flex h-12 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Talk to the team
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
