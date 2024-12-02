"use client";

import React, { useState } from "react";
import { DebugMessage } from "./types";
import MarkdownMessageDisplay from "../chat/display/markdown";

interface DebugMessageProps {
  message: DebugMessage;
  messageIndex: number;
}

const DebugMessageDisplay: React.FC<DebugMessageProps> = ({
  message,
  messageIndex,
}) => {
  const [collapsed, setCollapsed] = useState(
    message.role === "system" || message.role === "user"
  );

  return (
    <div
      key={"message" + messageIndex}
      className={`flex gap-2 flex-col ${
        message.role === "system"
          ? "bg-background_alt"
          : message.role === "user"
          ? "bg-foreground"
          : "bg-foreground_alt"
      } p-5 rounded-lg shadow-lg max-h-[500px] overflow-y-auto`}
    >
      <button
        className="w-full flex items-start justify-start"
        onClick={() => setCollapsed((prev) => !prev)}
      >
        <p
          className={`font-bold text-sm ${
            message.role === "system"
              ? "text-accent"
              : message.role === "user"
              ? "text-primary"
              : "text-highlight"
          }`}
        >
          {message.role}
        </p>
      </button>
      {!collapsed && <MarkdownMessageDisplay text={message.content} />}
    </div>
  );
};

export default DebugMessageDisplay;
