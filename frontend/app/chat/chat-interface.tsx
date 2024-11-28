"use client";

import React, { useEffect, useState, useRef } from "react";

import { Conversation, DecisionTreeNode, Query } from "../types";

import QueryInput from "./user-input";
import MessageDisplay from "./display/message-display";
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
  handleQuery: (query: string, conversationId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  currentConversation,
  conversations,
  toggleCollectionEnabled,
  handleQuery,
  routerChangeCollection,
}) => {
  const [currentQuery, setCurrentQuery] = useState<{
    [key: string]: Query;
  }>({});
  const [currentStatus, setCurrentStatus] = useState<string>("");

  const [mode, setMode] = useState<"chat" | "flow">("chat");
  const [currentTrees, setCurrentTrees] = useState<DecisionTreeNode[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setCurrentQuery(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.queries || {}
        : {}
    );
    setCurrentStatus(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.current || ""
        : ""
    );
    setCurrentTrees(
      currentConversation && conversations.length > 0
        ? conversations.find((c) => c.id === currentConversation)?.tree || []
        : []
    );
  }, [currentConversation, conversations]);

  useEffect(() => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  }, [currentQuery, currentStatus]);

  const handleSendQuery = (query: string) => {
    handleQuery(query, currentConversation);
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
        <div className="flex flex-col w-full justify-center items-center">
          <div className="flex flex-col overflow-scroll w-[60vw] h-[90vh]">
            {Object.entries(currentQuery)
              .sort((a, b) => a[1].index - b[1].index)
              .map(([queryId, query], index, array) => (
                <MessageDisplay
                  key={queryId}
                  messages={query.messages}
                  routerChangeCollection={routerChangeCollection}
                  _collapsed={index !== array.length - 1}
                  messagesEndRef={messagesEndRef}
                />
              ))}
            {!(Object.keys(currentQuery).length === 0) && (
              <div>
                <hr className="w-full border-t border-background my-4 mb-28" />
              </div>
            )}
          </div>
          <div className="w-full justify-center items-center flex">
            <QueryInput
              query_length={Object.keys(currentQuery).length}
              currentStatus={currentStatus}
              handleSendQuery={handleSendQuery}
            />
          </div>
        </div>
      ) : (
        <ReactFlowProvider>
          <FlowDisplay currentTrees={currentTrees} />
        </ReactFlowProvider>
      )}
    </div>
  );
};

export default ChatInterface;
