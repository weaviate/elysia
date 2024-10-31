"use client";

import Sidebar from "./navigation/sidebar";
import ChatInterface from "./chat/chat_interface";
import { useState } from "react";

export default function Home() {
  const [page, setPage] = useState<"home" | "data-explorer">("home");

  const handlePageChange = (_p: "home" | "data-explorer") => {
    setPage(_p);
  };

  return (
    <div className="w-full flex">
      <Sidebar handlePageChange={handlePageChange} page={page} />
      <ChatInterface />
    </div>
  );
}
