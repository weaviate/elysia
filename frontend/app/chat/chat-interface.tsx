"use client";

import React, { useEffect, useState } from "react";

import { Conversation, DecisionTreeNode, Message } from "../types";

import { v4 as uuidv4 } from "uuid";

import QueryInput from "./user-input";
import MessageDisplay from "./message-display";
import { BsChatFill } from "react-icons/bs";
import { RiFlowChart } from "react-icons/ri";
import FlowDisplay from "./flow-display";
import { ReactFlowProvider } from "@xyflow/react";
import SelectDropdown from "../navigation/select-dropdown";

interface ChatInterfaceProps {
  currentConversation: string;
  routerChangeCollection: (collection_id: string) => void;
  conversations: Conversation[];
  toggleCollectionEnabled: (
    collection_id: string,
    conversationId: string
  ) => void;
  addMessageToConversation: (
    message: Message[],
    conversationId: string
  ) => void;
  handleQuery: (query: string, conversationId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  currentConversation,
  conversations,
  toggleCollectionEnabled,
  addMessageToConversation,
  handleQuery,
  routerChangeCollection,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string>("");

  const [mode, setMode] = useState<"chat" | "flow">("chat");
  const [currentTree, setCurrentTree] = useState<DecisionTreeNode | null>(null);

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
    setCurrentTree(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.tree || null
        : null
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
      <div className="flex w-full justify-between items-center p-6">
        <div className="flex flex-col gap-2">
          <SelectDropdown
            title="Collections"
            selections={
              conversations.find((c) => c.id === currentConversation)
                ?.enabled_collections || {}
            }
            toggleOption={toggleCollectionEnabled}
            currentConversation={currentConversation}
          />
        </div>
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
            routerChangeCollection={routerChangeCollection}
          />
          <QueryInput messages={messages} handleSendQuery={handleSendQuery} />
        </>
      ) : (
        <ReactFlowProvider>
          <FlowDisplay currentTree={currentTree} />
        </ReactFlowProvider>
      )}
    </div>
  );
};

export default ChatInterface;
