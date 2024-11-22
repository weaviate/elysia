"use client";

import React from "react";
import MarkdownMessageDisplay from "./markdown";

interface WarningDisplayProps {
  warning: string;
}

const WarningDisplay: React.FC<WarningDisplayProps> = ({ warning }) => {
  return (
    <div className="w-full flex flex-col justify-start items-start ">
      <div className="max-w-3/5">
        <div className="flex flex-grow justify-start items-center gap-2 chat-animation bg-warning p-4 rounded-lg">
          <MarkdownMessageDisplay text={warning} />
        </div>
      </div>
    </div>
  );
};

export default WarningDisplay;
