"use client";

import React, { useState } from "react";
import { FaQuestionCircle } from "react-icons/fa";
import { IoArrowUpCircleSharp } from "react-icons/io5";
import { Message } from "../types";
import { IoDocumentText } from "react-icons/io5";
import { FaGithub } from "react-icons/fa";

interface QueryInputProps {
  handleSendQuery: (query: string) => void;
  messages: Message[];
}

const QueryInput: React.FC<QueryInputProps> = ({
  handleSendQuery,
  messages,
}) => {
  const width_control = messages.length == 0 ? "w-[40vw]" : "w-[60vw]";

  const [query, setQuery] = useState("");

  return (
    <div
      className={`fixed shadow-xl ${
        messages.length === 0 ? "top-1/2 -translate-y-1/2" : "bottom-8"
      } gap-4 flex items-center justify-center flex-col transition-all duration-300 ${width_control}`}
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
          messages.length === 0 ? "rounded-xl" : "rounded-full"
        } p-2 border border-foreground bg-background text-primary placeholder:text-secondary`}
      >
        <div
          className={`flex gap-2 w-full bg-background_alt ${
            messages.length === 0
              ? "rounded-xl items-end"
              : "rounded-full items-center"
          } p-2`}
        >
          <textarea
            placeholder={
              messages.length != 0
                ? "Ask a follow up question..."
                : "Elysia will search through your data..."
            }
            className={`w-full p-2 bg-transparent outline-none text-xs resize-none ${
              messages.length === 0
                ? "h-[20vh] rounded-xl"
                : "h-[3vh] rounded-full"
            }`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                handleSendQuery(query);
                setQuery("");
              }
            }}
          />
          <button
            className="btn-round text-secondary rounded-full"
            onClick={() => {
              handleSendQuery(query);
              setQuery("");
            }}
          >
            <IoArrowUpCircleSharp size={16} />
          </button>
        </div>
      </div>
      {messages.length == 0 && (
        <div className="grid grid-cols-2 w-full gap-3">
          <button
            onClick={() => {
              handleSendQuery("What is Elysia?");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaQuestionCircle size={16} />
            <p>What is Elysia?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("What is Verba?");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaQuestionCircle size={16} />
            <p>What is Verba?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("Summarize the last 10 GitHub Tickets");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <IoDocumentText size={16} />
            <p>Summarize the last 10 GitHub Tickets</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("Return all GitHub Tickets");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaGithub size={16} />
            <p>Return all GitHub Tickets</p>
          </button>
        </div>
      )}
    </div>
  );
};

export default QueryInput;
