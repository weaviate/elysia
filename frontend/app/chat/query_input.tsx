"use client";

import React, { useState } from "react";
import { RiSendPlane2Fill } from "react-icons/ri";

interface QueryInputProps {
  query: string;
  handleQueryChange: (query: string) => void;
  handleSendQuery: () => void;
}

const QueryInput: React.FC<QueryInputProps> = ({
  query,
  handleQueryChange,
  handleSendQuery,
}) => {
  return (
    <div className="w-full flex gap-2 rounded-full p-2 border border-foreground text-primary placeholder:text-secondary">
      <div className="flex gap-2 w-full items-center bg-background_alt rounded-full p-2">
        <input
          type="text"
          placeholder="Ask a follow up question..."
          className="w-full p-2 rounded-full bg-transparent outline-none text-xs"
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
  );
};

export default QueryInput;
