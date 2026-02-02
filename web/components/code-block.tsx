interface CodeBlockProps {
  code: string;
  lang: string;
  filename?: string;
}

export function CodeBlock({ code, filename }: CodeBlockProps) {
  // Shiki has been removed to resolve build errors.
  // Rendering code as plain text for now.

  return (
    <div className="my-6 overflow-hidden rounded-lg border bg-[#22272e] dark:border-zinc-800">
      {filename && (
        <div className="flex items-center justify-between bg-[#2d333b] px-4 py-2 border-b border-[#444c56]">
          <span className="text-xs font-mono text-zinc-300">{filename}</span>
        </div>
      )}
      <div className="p-4 overflow-x-auto">
        <pre className="!bg-transparent !p-0 !m-0 font-mono text-sm text-zinc-100">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
}
