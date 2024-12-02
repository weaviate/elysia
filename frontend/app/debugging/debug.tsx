"use client";

import React, { useEffect, useState } from "react";
import { DebugMessage, DebugResponse } from "./types";
import MarkdownMessageDisplay from "../chat/display/markdown";
import { Conversation } from "../types";

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
              <div
                key={key + "chat" + chatIndex}
                className="flex flex-col gap-6 p-4"
              >
                <p className="text-secondary text-sm">
                  ({chat.length}) Chat {chatIndex + 1}
                </p>
                {chat.map((message: DebugMessage, messageIndex: number) => (
                  <div
                    key={key + "chat" + chatIndex + "message" + messageIndex}
                    className={`flex gap-2 flex-col ${
                      message.role === "system"
                        ? "bg-background_alt"
                        : "bg-foreground"
                    } p-5 rounded-lg shadow-lg max-h-[500px] overflow-y-auto`}
                  >
                    <p
                      className={`font-bold text-sm ${
                        message.role === "system"
                          ? "text-accent"
                          : "text-highlight"
                      }`}
                    >
                      {message.role}
                    </p>
                    <MarkdownMessageDisplay text={message.content} />
                  </div>
                ))}
                <hr className="w-full border-t border-secondary my-4" />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default DebugView;
