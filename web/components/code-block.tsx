interface CodeBlockProps {
  code: string;
  lang: string;
  filename?: string;
}

export async function CodeBlock({ code, lang, filename }: CodeBlockProps) {
  const { codeToHtml } = await import("shiki");
  const html = await codeToHtml(code, {
    lang,
    themes: {
      light: "github-light",
      dark: "github-dark-dimmed",
    },
  });

  return (
    <div className="my-6 overflow-hidden rounded-lg border bg-[#22272e] dark:border-zinc-800">
      {filename && (
        <div className="flex items-center justify-between bg-[#2d333b] px-4 py-2 border-b border-[#444c56]">
          <span className="text-xs font-mono text-zinc-300">{filename}</span>
        </div>
      )}
      <div
        className="p-4 overflow-x-auto text-sm [&>pre]:!bg-transparent [&>pre]:p-0 [&>pre]:m-0 font-mono"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
