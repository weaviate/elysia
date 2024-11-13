"use server";

import { ErrorResponse, NERResponse, TitleResponse } from "../types";

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
  console.log(collection_names, remove_data, conversation_id, user_id);
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
