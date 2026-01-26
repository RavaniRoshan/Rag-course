import { getModule, getModules } from "@/lib/content";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { CodeBlock } from "@/components/code-block";
import { notFound } from "next/navigation";
import { Metadata } from "next";

export async function generateStaticParams() {
  const modules = await getModules();
  return modules.map((m) => ({
    slug: m.slug,
  }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const module = await getModule(slug);
  if (!module) return { title: 'Not Found' };

  return {
    title: `${module.title} - RAG Course`,
  };
}

export default async function ModulePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const module = await getModule(slug);

  if (!module) {
    notFound();
  }

  // Remove the H1 from markdown if it exists to avoid duplication with our own H1
  // Simple heuristic: if markdown starts with # Title, remove it.
  let content = module.markdown;
  const titleRegex = /^#\s+.+\n/;
  if (titleRegex.test(content)) {
    content = content.replace(titleRegex, '');
  }

  return (
    <div className="space-y-10 pb-20">
      <div>
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl mb-8">
            {module.title}
        </h1>
        <MarkdownRenderer content={content} />
      </div>

      {module.codeFiles.length > 0 && (
        <div className="border-t pt-10">
          <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0 mb-6">
            Code Examples
          </h2>
          <div className="space-y-8">
            {module.codeFiles.map((file) => (
              <div key={file.name}>
                <CodeBlock
                  code={file.content}
                  lang="python"
                  filename={file.name}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
