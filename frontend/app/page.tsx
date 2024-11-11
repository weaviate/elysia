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
  const [mode, setMode] = useState<"home" | "data-explorer">("home");
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
    toggleMessageCollapsed,
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
    pageUp,
    pageDown,
    pageUpMax,
    pageSize,
    pageDownMax,
    maxPage,
    page,
  } = useCollections();

  useEffect(() => {
    setConversations([initialConversation]);
    setCurrentConversation(initialConversation.id);
    generateIdFromIp().then((id) => setId(id));
  }, []);

  const handleModeChange = (_p: "home" | "data-explorer") => {
    setMode(_p);
  };

  const handleQuery = (query: string, conversationId: string) => {
    sendQuery(id || "", query, conversationId);
    setConversationTitle(query, conversationId);
  };

  return (
    <div className="w-full flex">
      <Sidebar
        handleModeChange={handleModeChange}
        mode={mode}
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
      {mode === "home" && currentConversation && (
        <ChatInterface
          currentConversation={currentConversation || ""}
          conversations={conversations}
          addMessageToConversation={addMessageToConversation}
          handleQuery={handleQuery}
          toggleMessageCollapsed={toggleMessageCollapsed}
        />
      )}
      {mode === "data-explorer" && (
        <DataExplorer
          collectionData={collectionData}
          collectionLoading={loadingCollection}
          collection={
            collections.find((c) => c.name === selectedCollection) || null
          }
          collectionName={selectedCollection || ""}
          page={page}
          pageUp={pageUp}
          pageDown={pageDown}
          pageUpMax={pageUpMax}
          pageDownMax={pageDownMax}
          maxPage={maxPage}
          pageSize={pageSize}
        />
      )}
    </div>
  );
}
