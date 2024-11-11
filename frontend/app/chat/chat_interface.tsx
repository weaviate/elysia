"use client";

import React, { useState } from "react";
import { RiSendPlane2Fill } from "react-icons/ri";

import { Conversation, Message } from "../types";

import QueryInput from "./query_input";
import MessageDisplay from "./message_display";

interface ChatInterfaceProps {
  currentConversation: string;
  toggleMessageCollapsed: (conversationId: string, message_id: string) => void;
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
  toggleMessageCollapsed,
  addMessageToConversation,
  handleQuery,
}) => {
  const messages =
    currentConversation && conversations.length > 0
      ? conversations.find((c) => c.id === currentConversation)?.messages || []
      : [];

  const current_status =
    currentConversation && conversations.length > 0
      ? conversations.find((c) => c.id === currentConversation)?.current || ""
      : "";

  const handleSendQuery = (query: string) => {
    if (query.trim() === "") return;
    const trimmedQuery = query.trim();
    const newMessage: Message = {
      type: "User",
      conversation_id: currentConversation,
      payload: {
        type: "text",
        metadata: {},
        objects: [trimmedQuery],
      },
    };
    addMessageToConversation([newMessage], currentConversation);
    handleQuery(trimmedQuery, currentConversation);
  };

  return (
    <div className="h-screen flex flex-col items-center justify-center flex-grow">
      <MessageDisplay
        messages={messages}
        current_status={current_status}
        toggleMessageCollapsed={toggleMessageCollapsed}
      />
      <QueryInput messages={messages} handleSendQuery={handleSendQuery} />
    </div>
  );
};

export default ChatInterface;
