import { useState } from "react";
import { Conversation, initialConversation, Message } from "../types";
import { v4 as uuidv4 } from "uuid";

import { handleConversationTitleGeneration } from "./api";

export function useConversations(id: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<string | null>(
    null
  );

  const addConversation = () => {
    const newConversation = { ...initialConversation, id: uuidv4() };
    setConversations([...(conversations || []), newConversation]);
    setCurrentConversation(newConversation.id);
  };

  const removeConversation = (id: string) => {
    if (currentConversation === id) {
      setCurrentConversation(null);
    }
    setConversations(conversations?.filter((c) => c.id !== id));
  };

  const selectConversation = (id: string) => {
    setCurrentConversation(id);
  };

  const setConversationStatus = (status: string, conversationId: string) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          return { ...c, current: status };
        }
        return c;
      })
    );
  };

  const setConversationTitle = (title: string, conversationId: string) => {
    handleConversationTitleGeneration(title).then((data) => {
      setConversations((prevConversations) =>
        prevConversations.map((c) => {
          if (c.id === conversationId && c.name === "New Conversation") {
            return { ...c, name: data.title };
          }
          return c;
        })
      );
    });
  };

  const setAllConversationStatuses = (status: string) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => ({ ...c, current: status }))
    );
  };

  const addMessageToConversation = (
    messages: Message[],
    conversationId: string
  ) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          return { ...c, messages: [...(c.messages || []), ...messages] };
        }
        return c;
      })
    );
  };

  const toggleMessageCollapsed = (
    conversationId: string,
    message_id: string
  ) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          return {
            ...c,
            messages: c.messages.map((m) =>
              m.id === message_id ? { ...m, collapsed: !m.collapsed } : m
            ),
          };
        }
        return c;
      })
    );
  };

  return {
    setConversations,
    setCurrentConversation,
    conversations,
    currentConversation,
    addConversation,
    removeConversation,
    selectConversation,
    addMessageToConversation,
    setConversationStatus,
    setAllConversationStatuses,
    toggleMessageCollapsed,
    setConversationTitle,
  };
}
