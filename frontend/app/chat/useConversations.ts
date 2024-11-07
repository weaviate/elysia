import { useState } from "react";
import { Conversation, initialConversation, Message } from "../types";
import { v4 as uuidv4 } from "uuid";

export function useConversations(id: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<string>();

  const addConversation = () => {
    const newConversation = { ...initialConversation, id: uuidv4() };
    setConversations([...(conversations || []), newConversation]);
    setCurrentConversation(newConversation.id);
  };

  const removeConversation = (id: string) => {
    setConversations(conversations?.filter((c) => c.id !== id));
    if (currentConversation === id) {
      setCurrentConversation(initialConversation.id);
    }
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

  const addMessageToConversation = (
    messages: Message[],
    conversationId: string
  ) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          console.log("Adding message to conversation", messages.length);
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
  };
}
