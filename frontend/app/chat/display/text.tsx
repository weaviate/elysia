"use client";

import { TextPayload } from "@/app/types";
import MarkdownMessageDisplay from "./markdown";

interface TextDisplayProps {
  payload: TextPayload[];
}

const TextDisplay: React.FC<TextDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-col justify-start items-start">
      {payload.map((text, idx) => (
        <div key={idx} className="chat-animation">
          <MarkdownMessageDisplay text={text.text} />
        </div>
      ))}
    </div>
  );
};

export default TextDisplay;
