import { DebugResponse } from "./types";

export async function getDebug(user_id: string, conversation_id: string) {
  const res = await fetch(`http://localhost:8000/api/debug`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_id, conversation_id }),
  });
  const data: DebugResponse = await res.json();
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}
