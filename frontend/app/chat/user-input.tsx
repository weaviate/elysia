"use client";

import React, { useState } from "react";
import { FaCircle, FaQuestionCircle } from "react-icons/fa";
import { IoArrowUpCircleSharp } from "react-icons/io5";
import { IoDocumentText } from "react-icons/io5";
import { FaGithub } from "react-icons/fa";
import { IoChatbubble } from "react-icons/io5";
import { PiPantsFill } from "react-icons/pi";
import { GiFishMonster } from "react-icons/gi";

interface QueryInputProps {
  handleSendQuery: (query: string) => void;
  query_length: number;
  currentStatus: string;
}

const QueryInput: React.FC<QueryInputProps> = ({
  handleSendQuery,
  query_length,
  currentStatus,
}) => {
  const width_control = query_length == 0 ? "w-[40vw]" : "w-[60vw]";

  const [query, setQuery] = useState("");

  return (
    <div
      className={`fixed ${
        query_length === 0 ? "top-1/2 -translate-y-1/2" : "bottom-8"
      } gap-4 flex items-center justify-center flex-col transition-all duration-300 ${width_control}`}
    >
      <p
        className={`text-2xl ${
          query_length === 0 ? "opacity-100" : "opacity-0"
        } transition-all duration-300 ease-in-out font-bold text-white`}
      >
        Ask anything!
      </p>
      {currentStatus != "" && (
        <div className="w-full flex justify-start items-center gap-2">
          <FaCircle className="text-secondary text-sm pulsing" />
          <p className="text-sm shine">{currentStatus}</p>
        </div>
      )}
      <div
        className={`w-full flex gap-2 ${
          query_length === 0 ? "rounded-xl" : "rounded-full"
        } p-2 border border-foreground bg-background text-primary placeholder:text-secondary`}
      >
        <div
          className={`flex gap-2 w-full bg-background_alt ${
            query_length === 0
              ? "rounded-xl items-end"
              : "rounded-full items-center"
          } p-2`}
        >
          <textarea
            placeholder={
              query_length != 0
                ? "Ask a follow up question..."
                : "Elysia will search through your data..."
            }
            className={`w-full p-2 bg-transparent outline-none text-xs resize-none ${
              query_length === 0
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
      {query_length == 0 && (
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
          <button
            onClick={() => {
              handleSendQuery("What was Edward's last conversation?");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <IoChatbubble size={16} />
            <p>What was the last conversation of Edward?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("To whom did Edward last message?");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <IoChatbubble size={16} />
            <p>To whom did Edward last message?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("I'm looking for green pants");
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <PiPantsFill size={16} />
            <p>I'm looking for green pants</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "I'm into Shrekcore, what are the best products?"
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <GiFishMonster size={16} />
            <p>I'm into Shrekcore, what are the best products?</p>
          </button>
        </div>
      )}
    </div>
  );
};

export default QueryInput;
