"use server";

import {
  DecisionTreeNode,
  DecisionTreePayload,
  ErrorResponse,
  Message,
  NERResponse,
  ObjectRelevancyPayload,
  TitleResponse,
} from "../types";

export async function handleNamedEntityRecognition(text: string) {
  const res = await fetch(`http://localhost:8000/api/ner`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });
  const data: NERResponse = await res.json();
  return data;
}

export async function handleConversationTitleGeneration(text: string) {
  const res = await fetch(`http://localhost:8000/api/title`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });
  const data: TitleResponse = await res.json();
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}

export async function setCollectionEnabled(
  collection_names: string[],
  remove_data: boolean,
  conversation_id: string,
  user_id: string
) {
  const res = await fetch(`http://localhost:8000/api/set_collections`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      collection_names,
      remove_data,
      conversation_id,
      user_id,
    }),
  });
  const data: ErrorResponse = await res.json();
  if (data.error) {
    throw new Error(data.error);
  }
}

export async function getDecisionTree(
  user_id: string,
  conversation_id: string
) {
  const res = await fetch(`http://localhost:8000/api/initialise_tree`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_id, conversation_id }),
  });
  const data: DecisionTreePayload = await res.json();
  const resetChoosenBlocked = (node: DecisionTreeNode) => {
    node.choosen = false;
    node.blocked = false;

    if (node.options) {
      Object.values(node.options).forEach((option) => {
        resetChoosenBlocked(option);
      });
    }
  };
  resetChoosenBlocked(data.tree);
  data.tree.choosen = true;
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}

export async function retrieveObjectRelevancy(
  user_id: string,
  conversation_id: string,
  query_id: string,
  objects: Message[]
) {
  const res = await fetch(`http://localhost:8000/api/object_relevance`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_id, conversation_id, query_id, objects }),
  });
  const data: ObjectRelevancyPayload = await res.json();
  console.log(data);
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}
