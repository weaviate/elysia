"use client";

import Sidebar from "./navigation/sidebar";
import ChatInterface from "./chat/chat_interface";
import { useEffect, useState } from "react";

import { generateIdFromIp } from "./util";

import { initialConversation } from "./types";
import { useConversations } from "./chat/useConversations";
import { useSocket } from "./chat/useSocket";
import { useCollections } from "./explorer/useCollections";
import DataExplorer from "./explorer/data_explorer";

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
    setConversationTitle,
    addMessageToConversation,
  } = useConversations(id || "");

  const { socketOnline, sendQuery } = useSocket(
    addMessageToConversation,
    setConversationStatus,
    setAllConversationStatuses
  );

  const {
    collections,
    fetchCollections,
    selectedCollection,
    selectCollection,
    collectionData,
    loadingCollection,
    loadingCollections,
  } = useCollections();

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
    setConversationTitle(query, conversationId);
  };

  return (
    <div className="w-full flex">
      <Sidebar
        handlePageChange={handlePageChange}
        page={page}
        fetchCollections={fetchCollections}
        selectCollection={selectCollection}
        selectedCollection={selectedCollection}
        collections={collections}
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
      {page === "data-explorer" && (
        <DataExplorer
          collectionData={collectionData}
          collectionLoading={loadingCollection}
          collection={
            collections.find((c) => c.name === selectedCollection) ||
            collections[0]
          }
          collectionName={selectedCollection || ""}
        />
      )}
    </div>
  );
}
