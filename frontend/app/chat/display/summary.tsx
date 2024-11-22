"use client";

import { SummaryPayload } from "@/app/types";
import MarkdownMessageDisplay from "./markdown";

interface SummaryDisplayProps {
  payload: SummaryPayload[];
}

const SummaryDisplay: React.FC<SummaryDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-col justify-start items-start">
      {payload.map((text, idx) => (
        <div
          key={idx}
          className="text-sm chat-animation text-white flex flex-col gap-4"
        >
          <p className="font-bold text-xl text-white">{text.title}</p>
          <MarkdownMessageDisplay text={text.text} />
        </div>
      ))}
    </div>
  );
};

export default SummaryDisplay;
