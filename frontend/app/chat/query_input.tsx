"use client";

import React, { useState } from "react";
import { RiSendPlane2Fill } from "react-icons/ri";
import { Message } from "../types";

interface QueryInputProps {
  query: string;
  handleQueryChange: (query: string) => void;
  handleSendQuery: () => void;
  messages: Message[];
}

const QueryInput: React.FC<QueryInputProps> = ({
  query,
  handleQueryChange,
  handleSendQuery,
  messages,
}) => {
  const width_control = messages.length == 0 ? "w-[40vw]" : "w-[60vw]";

  return (
    <div
      className={`gap-4 flex items-center justify-center flex-col transition-all duration-300 ${width_control}`}
    >
      <p
        className={`text-2xl ${
          messages.length === 0 ? "opacity-100" : "opacity-0"
        } transition-all duration-300 ease-in-out font-bold text-white`}
      >
        Ask anything!
      </p>
      <div
        className={`w-full flex gap-2 ${
          messages.length === 0 ? "rounded-md" : "rounded-full"
        } p-2 border border-foreground text-primary transition-all ease-in-out duration-300 placeholder:text-secondary`}
      >
        <div
          className={`flex gap-2 w-full items-center bg-background_alt  transition-all ease-in-out duration-300 ${
            messages.length === 0 ? "rounded-md" : "rounded-full"
          } p-2`}
        >
          <input
            type="textarea"
            placeholder="Ask a follow up question..."
            className={`w-full p-2 bg-transparent outline-none text-xs transition-all ease-in-out duration-300 ${
              messages.length === 0 ? "h-40 rounded-md" : "h-10 rounded-full"
            }`}
            value={query}
            onChange={(e) => handleQueryChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSendQuery();
              }
            }}
          />
          <button
            className="btn-round text-secondary rounded-full"
            onClick={handleSendQuery}
          >
            <RiSendPlane2Fill size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default QueryInput;
