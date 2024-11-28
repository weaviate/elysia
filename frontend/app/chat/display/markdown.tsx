"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

interface MarkdownMessageDisplayProps {
  text: string;
}

const MarkdownMessageDisplay: React.FC<MarkdownMessageDisplayProps> = ({
  text,
}) => {
  return (
    <div className="flex flex-col flex-grow prose-img:hidden prose-strong:text-highlight prose-strong:shadow-sm prose-a:text-accent prose-a:font-extralight prose-strong:font-extrabold justify-start items-start text-wrap prose max-w-none prose-ol:text-primary prose-ol:text-sm prose-ol:font-light prose-headings:text-primary prose-headings:text-base prose-headings:font-bold prose-p:text-primary prose-p:text-sm prose-p:font-medium prose-ul:text-primary prose-ul:text-sm prose-ul:font-light prose-code:font-mono prose-code:text-primary prose-code:text-xs prose-code:bg-background prose-pre:bg-background prose-pre:p-8 prose-pre:text-sm prose-pre:font-light prose-code:p-1 prose-code:rounded-lg prose-pre:w-full prose:w-full">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownMessageDisplay;
