"use client";

import MarkdownMessageDisplay from "./markdown";

interface TextDisplayProps {
  payload: string[];
}

const TextDisplay: React.FC<TextDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-col justify-start items-start gap-2">
      {payload.map((text, idx) => (
        <div key={idx} className="text-sm chat-animation text-white">
          <MarkdownMessageDisplay text={text} />
        </div>
      ))}
    </div>
  );
};

export default TextDisplay;