"use client";

import React, { useEffect, useState } from "react";
import { DebugMessage, DebugResponse } from "./types";
import { Conversation } from "../types";
import DebugChat from "./debug-chat";

interface DebugViewProps {
  fetchDebug: (conversation_id: string) => Promise<DebugResponse>;
  currentConversation: string;
  conversations: Conversation[];
}

const DebugView: React.FC<DebugViewProps> = ({
  fetchDebug,
  currentConversation,
  conversations,
}) => {
  const [debug, setDebug] = useState<DebugResponse | null>(null);

  const updateDebug = async (conversation_id: string) => {
    const debug = await fetchDebug(conversation_id);
    setDebug(debug);
  };

  useEffect(() => {
    updateDebug(currentConversation);
  }, [currentConversation, conversations]);

  if (!debug) return null;

  return (
    <div className="w-full p-8 flex flex-col gap-4 items-center justify-start h-[90vh] overflow-y-auto">
      <div className="w-[60vw] flex flex-col gap-4">
        <p className="text-xs text-secondary">
          Conversation: {currentConversation}
        </p>
        {Object.keys(debug).map((key) => (
          <div key={key} className="flex gap-2 flex-col">
            <div className="flex gap-2">
              <p className="font-bold text-lg">
                ({debug[key].chat.length}) {key}
              </p>
              <p className="text-secondary text-lg">{debug[key].model}</p>
            </div>
            {debug[key].chat.map((chat: DebugMessage[], chatIndex: number) => (
              <DebugChat
                key={key + "chat" + chatIndex}
                chat={chat}
                chatIndex={chatIndex}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default DebugView;
