"use client";

import React, { useState } from "react";
import { DebugMessage } from "./types";
import DebugMessageDisplay from "./debug-message";

interface DebugChatProps {
  chat: DebugMessage[];
  chatIndex: number;
}

const DebugChat: React.FC<DebugChatProps> = ({ chat, chatIndex }) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div key={"chat" + chatIndex} className="flex flex-col gap-6 p-4">
      <button
        className="w-full flex items-start justify-start hover:text-primary transition-colors duration-200 text-secondary"
        onClick={() => setCollapsed((prev) => !prev)}
      >
        <p className=" text-sm">
          ({chat.length}) Chat {chatIndex + 1}
        </p>
      </button>
      {!collapsed &&
        chat.map((message: DebugMessage, messageIndex: number) => (
          <DebugMessageDisplay
            key={"message" + messageIndex}
            message={message}
            messageIndex={messageIndex}
          />
        ))}
      <hr className="w-full border-t border-secondary my-4" />
    </div>
  );
};

export default DebugChat;
