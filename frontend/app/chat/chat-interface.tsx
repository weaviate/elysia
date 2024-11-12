"use client";

import React, { useEffect, useState } from "react";

import { Conversation, Message } from "../types";

import { v4 as uuidv4 } from "uuid";

import QueryInput from "./user-input";
import MessageDisplay from "./message-display";

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
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string>("");

  useEffect(() => {
    console.log(currentConversation, conversations);
    setMessages(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.messages ||
            []
        : []
    );
    setCurrentStatus(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.current || ""
        : ""
    );
  }, [currentConversation, conversations]);

  const handleSendQuery = (query: string) => {
    if (query.trim() === "") return;
    const trimmedQuery = query.trim();
    const newMessage: Message = {
      type: "User",
      id: uuidv4(),
      collapsed: false,
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
    <div className="h-screen flex flex-col items-center justify-start flex-grow">
      <MessageDisplay
        messages={messages}
        current_status={currentStatus}
        toggleMessageCollapsed={toggleMessageCollapsed}
      />
      <QueryInput messages={messages} handleSendQuery={handleSendQuery} />
    </div>
  );
};

export default ChatInterface;
