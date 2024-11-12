"use server";

import { NERResponse, TitleResponse } from "../types";

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
