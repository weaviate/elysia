"use client";

import { TextPayload } from "@/app/types";
import MarkdownMessageDisplay from "./markdown";

interface TextDisplayProps {
  payload: TextPayload[];
}

const TextDisplay: React.FC<TextDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-col gap-2 items-start justify-start">
      {payload.map((text, idx) => (
        <span
          key={idx}
          className={`chat-animation inline text-xs ${
            idx === payload.length - 1 ? "text-primary" : "text-secondary"
          } text-wrap transition-colors duration-300`}
        >
          {text.text}
        </span>
      ))}
    </div>
  );
};

export default TextDisplay;
