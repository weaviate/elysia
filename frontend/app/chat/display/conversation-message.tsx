"use client";

import React, { useEffect, useState } from "react";
import MarkdownMessageDisplay from "./markdown";
import { ConversationMessage } from "@/app/types";

interface ConversationMessageProps {
  payload: ConversationMessage[];
}

const ConversationMessageDisplay: React.FC<ConversationMessageProps> = ({
  payload,
}) => {
  const formatDate = (date: string) => {
    const dateObj = new Date(date);
    return dateObj.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const [messageCollapsed, setMessageCollapsed] = useState(false);

  useEffect(() => {
    payload.length > 2 && setMessageCollapsed(true);
  }, [payload]);

  return (
    <div className="w-full flex flex-col flex-grow gap-5 ">
      {(messageCollapsed ? payload.slice(0, 2) : payload).map(
        (message, idx) => (
          <div
            key={`${idx}-${message.conversation_id}`}
            className={`flex w-full ${
              idx % 2 === 0
                ? "justify-start items-start"
                : "justify-end items-end"
            }`}
          >
            <div className="flex flex-col w-full max-w-[35vw] shadow-lg gap-3 bg-background_alt p-4 rounded-lg chat-animation text-primary">
              <p
                className={`text-secondary text-xs font-bold ${
                  idx % 2 === 0 ? "text-left" : "text-right w-full"
                }`}
              >
                {message.message_author}
              </p>
              <MarkdownMessageDisplay text={message.message_content} />
              <div
                className={`flex w-full gap-2 ${
                  idx % 2 === 0
                    ? "justify-end items-end"
                    : "justify-start items-start"
                }`}
              >
                <p className="text-secondary text-xs">
                  {formatDate(message.message_timestamp)}
                </p>
              </div>
            </div>
          </div>
        )
      )}
      {payload.length > 2 && (
        <div className="flex w-full justify-center items-center">
          <button
            className="btn  text-secondary text-xs items-center justify-center"
            onClick={() => setMessageCollapsed((prev) => !prev)}
          >
            {messageCollapsed
              ? "Show All " + payload.length + " Messages"
              : "Show Less"}
          </button>
        </div>
      )}
    </div>
  );
};

export default ConversationMessageDisplay;
