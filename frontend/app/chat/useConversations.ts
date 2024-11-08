import { useState } from "react";
import { Conversation, initialConversation, Message } from "../types";
import { v4 as uuidv4 } from "uuid";

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
  };
}
