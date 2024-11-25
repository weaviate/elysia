"use client";

import { TextPayload } from "@/app/types";
import MarkdownMessageDisplay from "./markdown";

interface TextDisplayProps {
  payload: TextPayload[];
}

const TextDisplay: React.FC<TextDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-wrap gap-1 justify-start items-start">
      {payload.map((text, idx) => (
        <div key={idx} className="chat-animation text-wrap">
          <MarkdownMessageDisplay text={text.text} />
        </div>
      ))}
    </div>
  );
};

export default TextDisplay;
