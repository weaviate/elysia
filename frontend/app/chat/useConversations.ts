import { useEffect, useState } from "react";
import {
  Collection,
  Conversation,
  DecisionTreeNode,
  initialConversation,
  Message,
  TreeUpdatePayload,
} from "../types";
import { v4 as uuidv4 } from "uuid";

import {
  getDecisionTree,
  handleConversationTitleGeneration,
  setCollectionEnabled,
} from "./api";

import { getCollections } from "../explorer/api";

export function useConversations(id: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<string | null>(
    null
  );

  const [collections, setCollections] = useState<Collection[]>([]);

  const addConversation = () => {
    const conversation_id = uuidv4();
    getDecisionTree(id, conversation_id).then((data) => {
      getCollections().then((collections) => {
        setCollections(collections);
        const newConversation = {
          ...initialConversation,
          id: conversation_id,
          tree: [data.tree],
          base_tree: data.tree,
          enabled_collections: collections.reduce(
            (acc, c) => ({ ...acc, [c.name]: true }),
            {}
          ),
        };
        setConversations([...(conversations || []), newConversation]);
        setCurrentConversation(newConversation.id);
      });
    });
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

  const updateTree = (tree_update_message: Message) => {
    const payload = tree_update_message.payload as TreeUpdatePayload;

    const findAndUpdateNode = (
      tree: DecisionTreeNode | null,
      base_tree: DecisionTreeNode | null
    ): DecisionTreeNode | null => {
      if (!tree) {
        return null;
      }

      // If this is the node we're looking for
      if (tree.id === payload.node && !tree.blocked) {
        // Update the specific option within tree.options where option.name === payload.decision
        const updatedOptions = Object.entries(tree.options).reduce(
          (acc, [key, option]) => {
            if (key === payload.decision) {
              acc[key] = {
                ...option,
                choosen: true,
                reasoning: payload.reasoning,
                options: payload.reset
                  ? base_tree
                    ? { base: base_tree }
                    : {}
                  : option.options || {},
              };
            } else {
              acc[key] = option;
            }
            return acc;
          },
          {} as { [key: string]: DecisionTreeNode }
        );
        return { ...tree, options: updatedOptions, blocked: true };
      } else if (tree.options && Object.keys(tree.options).length > 0) {
        // Recurse into options
        const updatedOptions = Object.entries(tree.options).reduce(
          (acc, [key, option]) => {
            const updatedNode = findAndUpdateNode(option, base_tree);
            if (updatedNode) {
              acc[key] = updatedNode;
            }
            return acc;
          },
          {} as { [key: string]: DecisionTreeNode }
        );
        return { ...tree, options: updatedOptions, blocked: true };
      } else {
        return tree;
      }
    };

    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === tree_update_message.conversation_id) {
          const trees = c.tree;
          const tree = trees[payload.tree_index];
          const updatedTree = findAndUpdateNode(tree, c.base_tree);

          const newTrees = [...(c.tree || [])];
          if (updatedTree) {
            newTrees[payload.tree_index] = updatedTree;
          }
          return { ...c, tree: newTrees };
        }
        return c;
      })
    );
  };

  const addTreeToConversation = (conversationId: string) => {
    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId && c.base_tree) {
          return {
            ...c,
            tree: [...c.tree, { ...c.base_tree }],
          };
        }
        return c;
      })
    );
  };

  const changeBaseToQuery = (conversationId: string, query: string) => {
    const treeIndex =
      conversations.find((c) => c.id === conversationId)?.tree?.length || 1;

    setConversations((prevConversations) =>
      prevConversations.map((c) => {
        if (c.id === conversationId) {
          const newTrees = [...c.tree];
          if (newTrees[treeIndex - 1]) {
            newTrees[treeIndex - 1] = {
              ...newTrees[treeIndex - 1],
              name: query,
            };
          }
          return {
            ...c,
            tree: newTrees,
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
        /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
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
    setConversationTitle,
    initializeEnabledCollections,
    toggleCollectionEnabled,
    updateTree,
    addTreeToConversation,
    changeBaseToQuery,
  };
}
