import { getCourseReadme } from "@/lib/content";
import { MarkdownRenderer } from "@/components/markdown-renderer";

export default async function IntroductionPage() {
  const readme = await getCourseReadme();

  return (
    <div className="space-y-8 pb-20 animate-in fade-in duration-500">
      <div className="space-y-2">
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl text-foreground">
            Introduction
        </h1>
        <p className="text-xl text-muted-foreground">
            Welcome to the RAG Course. Below is the overview of what you will learn.
        </p>
      </div>
      <MarkdownRenderer content={readme} />
    </div>
  );
}
