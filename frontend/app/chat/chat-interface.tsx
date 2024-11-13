"use client";

import React, { useEffect, useState } from "react";

import { Conversation, DecisionPayload, Message } from "../types";

import { v4 as uuidv4 } from "uuid";

import QueryInput from "./user-input";
import MessageDisplay from "./message-display";
import { BsChatFill } from "react-icons/bs";
import { RiFlowChart } from "react-icons/ri";
import FlowDisplay from "./flow-display";
import { ReactFlow, ReactFlowProvider } from "@xyflow/react";

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
  const [decisions, setDecisions] = useState<DecisionPayload[]>([]);

  const [mode, setMode] = useState<"chat" | "flow">("chat");

  useEffect(() => {
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
    setDecisions(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.decisions ||
            []
        : []
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
      <div className="flex w-full justify-end items-end p-6">
        <div className="flex gap">
          <button
            className={`btn btn-round ${
              mode === "chat" ? "text-primary" : "text-secondary"
            }`}
            onClick={() => setMode("chat")}
          >
            <BsChatFill size={14} />
          </button>
          <button
            className={`btn btn-round ${
              mode === "flow" ? "text-primary" : "text-secondary"
            }`}
            onClick={() => setMode("flow")}
          >
            <RiFlowChart size={14} />
          </button>
        </div>
      </div>
      {mode === "chat" ? (
        <>
          <MessageDisplay
            messages={messages}
            current_status={currentStatus}
            toggleMessageCollapsed={toggleMessageCollapsed}
          />
          <QueryInput messages={messages} handleSendQuery={handleSendQuery} />
        </>
      ) : (
        <ReactFlowProvider>
          <FlowDisplay decisions={decisions} />
        </ReactFlowProvider>
      )}
    </div>
  );
};

export default ChatInterface;
