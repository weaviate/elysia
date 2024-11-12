"use client";

import React from "react";
import { MdWarning } from "react-icons/md";
import MarkdownMessageDisplay from "./markdown";

interface WarningDisplayProps {
  payload: string[];
}

const WarningDisplay: React.FC<WarningDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex flex-col justify-start items-start ">
      <div className="max-w-3/5">
        {payload.map((warning, idx) => (
          <div
            key={`${idx}-${warning}`}
            className="flex flex-grow justify-start items-center gap-2 chat-animation text-warning "
          >
            <MdWarning />
            <MarkdownMessageDisplay text={warning} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default WarningDisplay;
