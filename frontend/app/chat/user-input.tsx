"use client";

import React, { useState } from "react";
import { FaCircle, FaQuestionCircle } from "react-icons/fa";
import { IoArrowUpCircleSharp, IoClose } from "react-icons/io5";
import { IoDocumentText } from "react-icons/io5";
import { FaGithub } from "react-icons/fa";
import { IoChatbubble } from "react-icons/io5";
import { PiPantsFill } from "react-icons/pi";
import { FaCloud } from "react-icons/fa";
import { FaFileContract } from "react-icons/fa";
import { RiFlowChart } from "react-icons/ri";
import { FaTrash } from "react-icons/fa";
import { FaMoneyBillWave } from "react-icons/fa";
import { FaHouseChimney } from "react-icons/fa6";

interface QueryInputProps {
  handleSendQuery: (query: string, route?: string, mimick?: boolean) => void;
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

  const [route, setRoute] = useState<string>("");
  const [mimick, setMimick] = useState<boolean>(false);
  const [showRoute, setShowRoute] = useState<boolean>(false);

  return (
    <div
      className={`fixed ${
        query_length === 0 ? "top-1/2 -translate-y-1/2" : "bottom-8"
      } gap-4 flex items-center justify-center flex-col transition-all duration-300 ${width_control}`}
    >
      <p
        className={`text-xl ${
          query_length === 0 ? "opacity-100" : "opacity-0"
        } transition-all duration-300 ease-in-out font-bold font-merriweather text-white`}
      >
        Ask anything!
      </p>
      {currentStatus != "" && (
        <div className="w-full flex justify-start items-center gap-2">
          <FaCircle className="text-secondary text-sm pulsing" />
          <p className="text-sm shine">{currentStatus}</p>
        </div>
      )}
      {showRoute && (
        <div className="w-full flex gap-2 bg-background_alt rounded-xl p-2 fade-in justify-between">
          <input
            className="flex-grow p-2 bg-transparent outline-none text-xs resize-none"
            value={route}
            placeholder="Enter a route: e.g. search/query/text_response"
            onChange={(e) => setRoute(e.target.value)}
          />
          <div className="flex gap-2">
            {mimick ? (
              <button
                className="btn text-accent"
                onClick={() => setMimick(false)}
              >
                <p className="text-xs">Disable Mimicking</p>
              </button>
            ) : (
              <button
                className="btn text-secondary"
                onClick={() => setMimick(true)}
              >
                <p className="text-xs">Enable Mimicking</p>
              </button>
            )}
            <button
              className="btn-round text-secondary rounded-full"
              onClick={() => setRoute("")}
            >
              <FaTrash size={12} />
            </button>
            <button
              className="btn-round text-secondary rounded-full"
              onClick={() => setShowRoute(false)}
            >
              <IoClose size={12} />
            </button>
          </div>
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
            className={`w-full p-2 bg-transparent outline-none text-sm resize-none leading-tight ${
              query_length === 0
                ? "h-[20vh] rounded-xl"
                : "h-[3vh] rounded-full flex items-center"
            }`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSendQuery(query, route, mimick);
                setQuery("");
              }
            }}
            style={{
              overflow: "hidden",
              paddingTop: query_length === 0 ? "8px" : "2px",
            }}
          />
          <div className="flex gap-1">
            <button
              className={`btn-round ${
                showRoute && !route
                  ? "text-primary"
                  : route
                  ? "text-accent"
                  : "text-secondary"
              } rounded-full`}
              onClick={() => setShowRoute(!showRoute)}
            >
              <RiFlowChart size={16} />
            </button>
            <button
              className="btn-round text-secondary rounded-full"
              onClick={() => {
                handleSendQuery(query, route, mimick);
                setQuery("");
              }}
            >
              <IoArrowUpCircleSharp size={16} />
            </button>
          </div>
        </div>
      </div>
      {query_length == 0 && (
        <div className="grid grid-cols-2 w-full gap-3">
          <button
            onClick={() => {
              handleSendQuery("What is Elysia?", route, mimick);
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaQuestionCircle size={16} />
            <p>What is Elysia?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("What is Verba?", route, mimick);
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaQuestionCircle size={16} />
            <p>What is Verba?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "Summarize the last 10 GitHub Tickets",
                route,
                mimick
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaGithub size={16} />
            <p>Summarize the last 10 GitHub Tickets</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "Aggregate all usernames that wrote issues",
                route,
                mimick
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <IoDocumentText size={16} />
            <p>Aggregate all usernames that wrote issues</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "What was Edward's last conversation?",
                route,
                mimick
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <IoChatbubble size={16} />
            <p>What was the last conversation of Edward?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "When was the highest wind speed?",
                route,
                mimick
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaCloud size={16} />
            <p>When was the highest wind speed?</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("I'm looking for green pants", route, mimick);
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <PiPantsFill size={16} />
            <p>{`I'm looking for green pants`}</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "How many Sale Agreements are there?",
                route,
                mimick
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaFileContract size={16} />
            <p>{`How many Sale Agreements are there?`}</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery("How high is Mark Robson salary?", route, mimick);
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaMoneyBillWave size={16} />
            <p>{`How high is Mark Robson salary?`}</p>
          </button>
          <button
            onClick={() => {
              handleSendQuery(
                "Where is Weaviate's entity located?",
                route,
                mimick
              );
            }}
            className="btn w-full bg-background_alt text-primary text-sm"
          >
            <FaHouseChimney size={16} />
            <p>{`Where is Weaviate's entity located?`}</p>
          </button>
        </div>
      )}
    </div>
  );
};

export default QueryInput;
