"use client";

import React, { useState } from "react";
import { RiSendPlane2Fill } from "react-icons/ri";

import { Conversation, Message } from "../types";

import QueryInput from "./query_input";
import MessageDisplay from "./message_display";

interface ChatInterfaceProps {
  currentConversation: string;
  conversations: Conversation[];
  addMessageToConversation: (
    message: Message[],
    conversationId: string
  ) => void;
  handleQuery: (query: string, conversationId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  currentConversation,
  conversations,
  addMessageToConversation,
  handleQuery,
}) => {
  const [query, setQuery] = useState("");
  const messages =
    currentConversation && conversations.length > 0
      ? conversations.find((c) => c.id === currentConversation)?.messages || []
      : [];

  const current_status =
    currentConversation && conversations.length > 0
      ? conversations.find((c) => c.id === currentConversation)?.current || ""
      : "";

  const handleQueryChange = (q: string) => {
    setQuery(q);
  };

  const handleSendQuery = () => {
    if (query.trim() === "") return;
    const newMessage: Message = {
      type: "User",
      conversation_id: currentConversation,
      payload: {
        type: "text",
        metadata: {},
        objects: [query],
      },
    };
    addMessageToConversation([newMessage], currentConversation);
    handleQuery(query, currentConversation);
    setQuery("");
  };

  return (
    <div className="h-screen flex flex-col items-center justify-center flex-grow">
      <MessageDisplay messages={messages} current_status={current_status} />
      <QueryInput
        query={query}
        messages={messages}
        handleQueryChange={handleQueryChange}
        handleSendQuery={handleSendQuery}
      />
    </div>
  );
};

export default ChatInterface;
