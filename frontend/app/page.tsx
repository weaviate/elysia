"use client";

import Sidebar from "@/app/navigation/sidebar";
import ChatInterface from "@/app/chat/chat-interface";
import { useEffect, useState } from "react";

import { generateIdFromIp } from "./util";

import { v4 as uuidv4 } from "uuid";
import { useConversations } from "./chat/useConversations";
import { useSocket } from "./chat/useSocket";
import { useCollections } from "./explorer/useCollections";
import DataExplorer from "./explorer/data-explorer";
import { useRouting } from "./navigation/useRouting";
import { Message } from "./types";
import { useDebug } from "./debugging/useDebug";

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
    conversations,
    currentConversation,
    addConversation,
    removeConversation,
    setConversationStatus,
    setAllConversationStatuses,
    selectConversation,
    setConversationTitle,
    updateTree,
    toggleCollectionEnabled,
    addTreeToConversation,
    addMessageToConversation,
    changeBaseToQuery,
    addQueryToConversation,
    creatingNewConversation,
  } = useConversations(id || "");

  const { socketOnline, sendQuery } = useSocket(
    addMessageToConversation,
    setConversationStatus,
    setAllConversationStatuses,
    updateTree,
    id || ""
  );

  const { fetchDebug } = useDebug(id || "");

  useEffect(() => {
    generateIdFromIp().then((id) => {
      setId(id);
      addConversation();
    });
  }, []);

  const handleQuery = (query: string, conversationId: string) => {
    if (query.trim() === "") return;
    const trimmedQuery = query.trim();
    const query_id = uuidv4();
    sendQuery(id || "", trimmedQuery, conversationId, query_id);
    changeBaseToQuery(conversationId, trimmedQuery);
    setConversationTitle(trimmedQuery, conversationId);
    addTreeToConversation(conversationId);
    addQueryToConversation(conversationId, trimmedQuery, query_id);
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
        creatingNewConversation={creatingNewConversation}
      />
      {mode === "home" && currentConversation && (
        <ChatInterface
          currentConversation={currentConversation || ""}
          conversations={conversations}
          toggleCollectionEnabled={toggleCollectionEnabled}
          handleQuery={handleQuery}
          routerChangeCollection={routerChangeCollection}
          fetchDebug={fetchDebug}
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
