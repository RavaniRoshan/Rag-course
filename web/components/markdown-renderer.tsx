import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Mermaid } from './mermaid';

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose dark:prose-invert max-w-none prose-headings:scroll-m-20 prose-headings:font-semibold prose-headings:tracking-tight prose-h1:text-4xl prose-h2:text-3xl prose-h3:text-2xl prose-p:leading-7 prose-ul:my-6 prose-ul:ml-6 prose-ul:list-disc [&_code]:bg-muted [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:font-mono [&_code]:text-sm [&_pre]:bg-muted [&_pre]:p-4 [&_pre]:rounded-lg [&_pre]:overflow-x-auto [&_pre_code]:bg-transparent [&_pre_code]:p-0 prose-code:before:content-none prose-code:after:content-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code(props) {
            const {children, className, ...rest} = props
            const match = /language-(\w+)/.exec(className || '')

            if (match && match[1] === 'mermaid') {
              return <Mermaid chart={String(children).replace(/\n$/, '')} />
            }

            return <code className={className} {...rest}>{children}</code>
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
