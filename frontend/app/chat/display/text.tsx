"use client";

import { TextPayload } from "@/app/types";
import MarkdownMessageDisplay from "./markdown";

interface TextDisplayProps {
  payload: TextPayload[];
}

const TextDisplay: React.FC<TextDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-col gap-4 items-start justify-start">
      {/* Show merged text for all except last item */}
      {payload.length > 1 && (
        <span
          key={payload
            .slice(0, -1)
            .map((item) => item.text)
            .join(" ")}
          className="text-secondary text-sm text-wrap  transition-colors duration-300 chat-animation"
        >
          {payload
            .slice(0, -1)
            .map((item) => item.text)
            .join(" ")}
        </span>
      )}
      {/* Show last item separately */}
      {payload.length > 0 && (
        <div
          className="fade-in flex w-full"
          key={payload[payload.length - 1].text}
        >
          <MarkdownMessageDisplay text={payload[payload.length - 1].text} />
        </div>
      )}
    </div>
  );
};

export default TextDisplay;
