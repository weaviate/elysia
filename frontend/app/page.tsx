"use client";

import { v4 as uuidv4 } from "uuid";

import Sidebar from "./navigation/sidebar";
import ChatInterface from "./chat/chat_interface";
import { useEffect, useState } from "react";

import { generateIdFromIp } from "./util";

import { Conversation, initialConversation, Message } from "./types";
import { useConversations } from "./chat/useConversations";
import { useSocket } from "./chat/useSocket";

export default function Home() {
  const [page, setPage] = useState<"home" | "data-explorer">("home");
  const [id, setId] = useState<string>();

  const {
    setConversations,
    setCurrentConversation,
    conversations,
    currentConversation,
    addConversation,
    removeConversation,
    setConversationStatus,
    setAllConversationStatuses,
    selectConversation,
    addMessageToConversation,
  } = useConversations(id || "");

  const { socketOnline, sendQuery } = useSocket(
    addMessageToConversation,
    setConversationStatus,
    setAllConversationStatuses
  );

  useEffect(() => {
    setConversations([initialConversation]);
    setCurrentConversation(initialConversation.id);
    generateIdFromIp().then((id) => setId(id));
  }, []);

  const handlePageChange = (_p: "home" | "data-explorer") => {
    setPage(_p);
  };

  const handleQuery = (query: string, conversationId: string) => {
    sendQuery(id || "", query, conversationId);
  };

  return (
    <div className="w-full flex">
      <Sidebar
        handlePageChange={handlePageChange}
        page={page}
        socketOnline={socketOnline}
        conversations={conversations}
        currentConversation={currentConversation || ""}
        addConversation={addConversation}
        removeConversation={removeConversation}
        selectConversation={selectConversation}
      />
      {page === "home" && currentConversation && (
        <ChatInterface
          currentConversation={currentConversation || ""}
          conversations={conversations}
          addMessageToConversation={addMessageToConversation}
          handleQuery={handleQuery}
        />
      )}
    </div>
  );
}
