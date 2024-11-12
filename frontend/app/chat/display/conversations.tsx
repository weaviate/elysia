"use client";

import React, { useEffect, useState } from "react";
import { ConversationMessage } from "@/app/types";
import ConversationMessageDisplay from "./conversation-message";

interface ConversationsDisplayProps {
  payload: ConversationMessage[][];
}

const ConversationsDisplay: React.FC<ConversationsDisplayProps> = ({
  payload,
}) => {
  const [conversationCollapsed, setConversationCollapsed] = useState(false);

  useEffect(() => {
    payload.length > 1 && setConversationCollapsed(true);
  }, [payload]);

  return (
    <div className="w-full flex flex-col justify-start items-start ">
      {(conversationCollapsed ? payload.slice(0, 1) : payload).map(
        (conversation, idx) => (
          <div
            className="w-full"
            key={`${idx}-${conversation[0].conversation_id}`}
          >
            <ConversationMessageDisplay
              key={`${idx}-${conversation[0].conversation_id}`}
              payload={conversation}
            />
            <hr className="w-full border-t border-secondary my-4" />
          </div>
        )
      )}
      {payload.length > 1 && (
        <div className="flex w-full justify-center items-center">
          <button
            className="btn bg-background text-primary text-xs items-center justify-center"
            onClick={() => setConversationCollapsed((prev) => !prev)}
          >
            {conversationCollapsed
              ? "Show All " + payload.length + " Conversations"
              : "Show Less"}
          </button>
        </div>
      )}
    </div>
  );
};

export default ConversationsDisplay;
