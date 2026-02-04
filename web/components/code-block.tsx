"use client"

import * as React from "react"
import { Check, Copy } from "lucide-react"

interface CodeBlockProps {
  code: string;
  lang?: string;
  filename?: string;
}

export function CodeBlock({ code, filename, lang }: CodeBlockProps) {
  const [isCopied, setIsCopied] = React.useState(false)

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setIsCopied(true)
      setTimeout(() => setIsCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy text: ", err)
    }
  }

  return (
    <div className="group relative my-6 overflow-hidden rounded-lg border bg-[#22272e] dark:border-zinc-800">
      {filename && (
        <div className="flex items-center justify-between bg-[#2d333b] px-4 py-2 border-b border-[#444c56]">
          <span className="text-xs font-mono text-zinc-300">{filename}</span>
          <button
            onClick={copyToClipboard}
            className="inline-flex items-center justify-center rounded-md p-1.5 text-zinc-400 hover:bg-[#444c56] hover:text-white transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:outline-none"
            aria-label={isCopied ? "Copied" : "Copy code"}
          >
            {isCopied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </button>
        </div>
      )}

      {!filename && (
        <button
          onClick={copyToClipboard}
          className="absolute top-3 right-3 z-10 inline-flex items-center justify-center rounded-md p-1.5 bg-[#2d333b] border border-[#444c56] text-zinc-400 hover:bg-[#444c56] hover:text-white transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:outline-none"
          aria-label={isCopied ? "Copied" : "Copy code"}
        >
          {isCopied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
        </button>
      )}

      <div className="p-4 overflow-x-auto">
        <pre className="!bg-transparent !p-0 !m-0 font-mono text-sm text-zinc-100">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  )
}
