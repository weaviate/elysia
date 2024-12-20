"use server";

import { CollectionPayload, CollectionData } from "../types";

import host from "../host";

export async function getCollections() {
  const res = await fetch(`http://${host}/api/collections`);
  const data: CollectionPayload = await res.json();
  if (data.error) {
    throw new Error(data.error);
  }
  return data.collections;
}

export async function getCollection(
  collection_name: string,
  page: number,
  pageSize: number
) {
  const res = await fetch(`http://${host}/api/get_collection`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ collection_name, page, pageSize }),
  });
  const data: CollectionData = await res.json();
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}
