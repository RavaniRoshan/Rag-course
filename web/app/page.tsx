import Link from "next/link";
import { ArrowRight, BookOpen, Layers, Terminal } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <section className="flex-1 flex flex-col items-center justify-center space-y-10 py-24 text-center md:py-32">
        <div className="space-y-4 max-w-3xl mx-auto px-4">
          <h1 className="text-4xl font-extrabold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Master Retrieval Augmented Generation
          </h1>
          <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
            A comprehensive, deep-dive course into building advanced RAG systems. From chunking strategies to agentic workflows.
          </p>
        </div>
        <div className="flex gap-4">
          <Link
            href="/introduction"
            className="inline-flex h-12 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            Start Course
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
      </section>

      <section className="container mx-auto px-4 py-12 md:py-24 grid gap-8 md:grid-cols-3">
        <div className="flex flex-col items-center text-center space-y-4 p-6 rounded-lg border bg-card text-card-foreground shadow-sm hover:shadow-md transition-shadow">
          <div className="p-3 rounded-full bg-primary/10 text-primary">
            <Layers className="h-6 w-6" />
          </div>
          <h3 className="text-xl font-bold">Deep Dive Modules</h3>
          <p className="text-muted-foreground">
            Explore detailed modules covering everything from Reranking to GraphRAG.
          </p>
        </div>
        <div className="flex flex-col items-center text-center space-y-4 p-6 rounded-lg border bg-card text-card-foreground shadow-sm hover:shadow-md transition-shadow">
           <div className="p-3 rounded-full bg-primary/10 text-primary">
            <Terminal className="h-6 w-6" />
          </div>
          <h3 className="text-xl font-bold">Code First</h3>
          <p className="text-muted-foreground">
            Practical implementation examples and code snippets for every concept.
          </p>
        </div>
        <div className="flex flex-col items-center text-center space-y-4 p-6 rounded-lg border bg-card text-card-foreground shadow-sm hover:shadow-md transition-shadow">
           <div className="p-3 rounded-full bg-primary/10 text-primary">
            <BookOpen className="h-6 w-6" />
          </div>
          <h3 className="text-xl font-bold">Visual Learning</h3>
          <p className="text-muted-foreground">
            Clear diagrams and visualizations to understand complex architectures.
          </p>
        </div>
      </section>
    </div>
  );
}
