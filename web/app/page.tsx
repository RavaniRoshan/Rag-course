import { getCourseReadme } from "@/lib/content";
import { MarkdownRenderer } from "@/components/markdown-renderer";

export default async function Home() {
  const readme = await getCourseReadme();

  return (
    <div className="space-y-8 pb-20">
      <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
        RAG Course
      </h1>
      <MarkdownRenderer content={readme} />
    </div>
  );
}
