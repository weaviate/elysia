"use client";

import ReactMarkdown from "react-markdown";

interface MarkdownMessageDisplayProps {
  text: string;
}

const MarkdownMessageDisplay: React.FC<MarkdownMessageDisplayProps> = ({
  text,
}) => {
  return (
    <div className="flex gap-8 flex-col flex-grow justify-start items-start text-wrap">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
};

export default MarkdownMessageDisplay;
