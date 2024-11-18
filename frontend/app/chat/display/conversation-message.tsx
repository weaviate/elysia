"use client";

import React, { useEffect, useState } from "react";
import MarkdownMessageDisplay from "./markdown";
import { ConversationMessage } from "@/app/types";
import CollectionDisplay from "./collection";

interface ConversationMessageProps {
  payload: ConversationMessage[];
}

const AUTHOR_COLORS = [
  "text-accent",
  "text-highlight",
  "text-warning",
  "text-error",
];

const ConversationMessageDisplay: React.FC<ConversationMessageProps> = ({
  payload,
}) => {
  const formatDate = (date: string) => {
    const dateObj = new Date(date);
    return dateObj.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    });
  };

  const [messageCollapsed, setMessageCollapsed] = useState(false);
  const [authorColors, setAuthorColors] = useState<Record<string, string>>({});
  const [authorPositions, setAuthorPositions] = useState<
    Record<string, number>
  >({});

  useEffect(() => {
    if (payload.length > 2) {
      setMessageCollapsed(true);
    }

    // Get unique authors
    const uniqueAuthors = Array.from(
      new Set(payload.map((msg) => msg.message_author))
    );

    // Create color and position assignments
    const colorMap: Record<string, string> = {};
    const positionMap: Record<string, number> = {};
    uniqueAuthors.forEach((author, index) => {
      const colorIndex = index % AUTHOR_COLORS.length;
      colorMap[author] = AUTHOR_COLORS[colorIndex];
      positionMap[author] = index;
    });

    setAuthorColors(colorMap);
    setAuthorPositions(positionMap);
  }, [payload]);

  return (
    <div className="w-full flex flex-col flex-grow gap-5 ">
      {(messageCollapsed ? payload.slice(0, 2) : payload).map(
        (message, idx) => (
          <div
            key={`${idx}-${message.conversation_id}`}
            className={`flex w-full ${
              authorPositions[message.message_author] % 2 === 0
                ? "justify-start items-start"
                : "justify-end items-end"
            }`}
          >
            <div className="flex flex-col w-full max-w-[35vw] shadow-lg gap-3 bg-background_alt p-4 rounded-lg chat-animation text-primary">
              <p
                className={`${
                  authorColors[message.message_author]
                } text-xs font-bold ${
                  authorPositions[message.message_author] % 2 === 0
                    ? "text-left"
                    : "text-right w-full"
                }`}
              >
                {message.message_author}
              </p>
              <MarkdownMessageDisplay text={message.message_content} />
              <div
                className={`flex w-full gap-2 ${
                  authorPositions[message.message_author] % 2 === 0
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
