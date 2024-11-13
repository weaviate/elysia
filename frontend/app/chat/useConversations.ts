import { useEffect, useState } from "react";
import {
  Collection,
  Conversation,
  DecisionPayload,
  initialConversation,
  Message,
} from "../types";
import { v4 as uuidv4 } from "uuid";

import { handleConversationTitleGeneration, setCollectionEnabled } from "./api";

export function useConversations(id: string, collections: Collection[]) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<string | null>(
    null
  );

  const addConversation = () => {
    const newConversation = {
      ...initialConversation,
      id: uuidv4(),
      enabled_collections: collections.reduce(
        (acc, c) => ({ ...acc, [c.name]: true }),
        {}
      ),
    };
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

  const addDecisionToConversation = (
    decision: DecisionPayload,
    conversationId: string
  ) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          return { ...c, decisions: [...(c.decisions || []), decision] };
        }
        return c;
      })
    );
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
    const conversation = conversations.find((c) => c.id === conversationId);
    if (!conversation || conversation.name !== "New Conversation") return;
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

  const initializeEnabledCollections = (
    collections: { [key: string]: boolean },
    collection_id: string
  ) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === collection_id) {
          return { ...c, enabled_collections: collections };
        }
        return c;
      })
    );
  };

  const toggleCollectionEnabled = (
    collection_id: string,
    conversationId: string
  ) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          const new_enabled_collections = {
            ...c.enabled_collections,
            [collection_id]: !c.enabled_collections[collection_id],
          };
          return {
            ...c,
            enabled_collections: new_enabled_collections,
          };
        }
        return c;
      })
    );
  };

  useEffect(() => {
    if (!collections) return;
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (
          !c.enabled_collections ||
          Object.keys(c.enabled_collections).length === 0
        ) {
          return {
            ...c,
            enabled_collections: collections.reduce(
              (acc, c) => ({ ...acc, [c.name]: true }),
              {}
            ),
          };
        }
        return c;
      })
    );
  }, [collections]);

  useEffect(() => {
    conversations.map((c) => {
      const active_collections = Object.entries(c.enabled_collections || [])
        .filter(([_, enabled]) => enabled)
        .map(([name]) => name);
      setCollectionEnabled(active_collections, false, c.id, id);
    });
  }, [conversations]);

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
    addDecisionToConversation,
    setConversationTitle,
    initializeEnabledCollections,
    toggleCollectionEnabled,
  };
}
