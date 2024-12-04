"use client";

import host from "./host";
export const getWebsocketHost = () => {
  return `ws://${host}/ws/query`;
};
