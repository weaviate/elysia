"use client";

import React, { useState } from "react";
import { RiSendPlane2Fill } from "react-icons/ri";

import { Message } from "../types";

import QueryInput from "./query_input";
import MessageDisplay from "./message_display";

interface ChatInterfaceProps {}

const ChatInterface: React.FC<ChatInterfaceProps> = ({}) => {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  const handleQueryChange = (q: string) => {
    setQuery(q);
  };

  const handleSendQuery = () => {
    setMessages([...messages, { user_message: query, system_message: null }]);
    setQuery("");
  };

  return (
    <div className="h-screen flex flex-col p-10 items-center justify-center flex-grow">
      <MessageDisplay messages={messages} />
      <div className="flex w-[50vw]">
        <QueryInput
          query={query}
          handleQueryChange={handleQueryChange}
          handleSendQuery={handleSendQuery}
        />
      </div>
    </div>
  );
};

export default ChatInterface;
