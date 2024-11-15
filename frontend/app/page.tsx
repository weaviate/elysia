"use client";

import Sidebar from "@/app/navigation/sidebar";
import ChatInterface from "@/app/chat/chat-interface";
import { useEffect, useState } from "react";

import { generateIdFromIp } from "./util";

import { initialConversation } from "./types";
import { useConversations } from "./chat/useConversations";
import { useSocket } from "./chat/useSocket";
import { useCollections } from "./explorer/useCollections";
import DataExplorer from "./explorer/data-explorer";
import { useRouting } from "./navigation/useRouting";

export default function Home() {
  const [mode, setMode] = useState<"home" | "data-explorer">("home");
  const [id, setId] = useState<string>();

  const handleModeChange = (_p: "home" | "data-explorer") => {
    setMode(_p);
  };

  const {
    collections,
    fetchCollections,
    selectedCollection,
    selectCollection,
    collectionData,
    loadingCollection,
    pageSize,
    setPage,
    maxPage,
    page,
  } = useCollections();

  const {
    routerChangeMode,
    routerChangeCollection,
    pageUp,
    pageDown,
    pageUpMax,
    pageDownMax,
    routerToLogin,
  } = useRouting(
    handleModeChange,
    mode,
    selectCollection,
    selectedCollection,
    collections,
    maxPage,
    page,
    setPage
  );

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
    addDecisionToConversation,
    toggleMessageCollapsed,
    toggleCollectionEnabled,
    addMessageToConversation,
  } = useConversations(id || "", collections);

  const { socketOnline, sendQuery } = useSocket(
    addMessageToConversation,
    setConversationStatus,
    setAllConversationStatuses,
    addDecisionToConversation
  );

  useEffect(() => {
    setConversations([initialConversation]);
    setCurrentConversation(initialConversation.id);
    generateIdFromIp().then((id) => setId(id));
  }, []);

  const handleQuery = (query: string, conversationId: string) => {
    sendQuery(id || "", query, conversationId);
    setConversationTitle(query, conversationId);
  };

  return (
    <div className="w-full flex">
      <Sidebar
        handleModeChange={routerChangeMode}
        mode={mode}
        routerToLogin={routerToLogin}
        fetchCollections={fetchCollections}
        selectCollection={routerChangeCollection}
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
          toggleCollectionEnabled={toggleCollectionEnabled}
          addMessageToConversation={addMessageToConversation}
          handleQuery={handleQuery}
          toggleMessageCollapsed={toggleMessageCollapsed}
          routerChangeCollection={routerChangeCollection}
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
