"use client";

import React, { useEffect, useRef } from "react";

import { Message, ResultPayload, Ticket } from "../types";

import UserMessageDisplay from "./user_message_display";
import MarkdownMessageDisplay from "./markdown_display";
import ErrorMessageDisplay from "./error_message_display";
import TicketMessageDisplay from "./ticket_display";
import { FaCircle } from "react-icons/fa6";
import { TfiMoreAlt } from "react-icons/tfi";

interface MessageDisplayProps {
  messages: Message[];
  current_status: string;
  toggleMessageCollapsed: (conversationId: string, message_id: string) => void;
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({
  messages,
  current_status,
  toggleMessageCollapsed,
}) => {
  const height_control = messages.length == 0 ? "h-[0px]" : "h-[80vh]";
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, current_status]);

  return (
    <div
      className={`w-[60vw] flex justify-start items-start mt-10 p-4 overflow-scroll transition-all duration-300 ${height_control}`}
    >
      <div className="flex flex-col gap-6 w-full">
        {messages.map((message, index) => (
          <div key={index + "message"} className="w-full flex">
            {message.type === "User" && (
              <div className="w-full flex flex-col justify-start items-start mt-8 ">
                <div className="max-w-3/5">
                  {(message.payload as ResultPayload).objects.map(
                    (text, idx) => (
                      <UserMessageDisplay
                        key={`${index}-${idx}`}
                        user_message={text as string}
                      />
                    )
                  )}
                </div>
              </div>
            )}
            {message.type === "result" && (
              <div className="w-full flex flex-col justify-start items-start ">
                <div className="w-full flex flex-col justify-start items-start gap-2">
                  {(message.payload as ResultPayload).type === "text" &&
                    (message.payload as ResultPayload).objects.map(
                      (text, idx) => (
                        <MarkdownMessageDisplay
                          key={`${index}-${idx}`}
                          markdown_message={text as string}
                        />
                      )
                    )}
                  {(message.payload as ResultPayload).type === "ticket" && (
                    <>
                      {(message.collapsed
                        ? (message.payload as ResultPayload).objects.slice(0, 3)
                        : (message.payload as ResultPayload).objects
                      ).map((ticket, idx) => (
                        <TicketMessageDisplay
                          key={`${index}-${idx}`}
                          ticket={ticket as Ticket}
                        />
                      ))}
                      {(message.payload as ResultPayload).objects.length >
                        3 && (
                        <div className="flex w-full justify-center items-center">
                          <button
                            className="btn w-1/5 bg-background text-primary text-xs items-center justify-center"
                            onClick={() => {
                              toggleMessageCollapsed(
                                message.conversation_id,
                                message.id || ""
                              );
                            }}
                          >
                            <TfiMoreAlt className="text-primary text-xs" />
                            {message.collapsed
                              ? "Show All " +
                                (message.payload as ResultPayload).objects
                                  .length
                              : "Show Less"}
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
            {message.type === "error" && (
              <div className="w-full flex flex-col justify-start items-start ">
                <div className="max-w-3/5">
                  {(message.payload as ResultPayload).objects.map(
                    (error, idx) => (
                      <ErrorMessageDisplay
                        key={`${index}-${idx}`}
                        error_message={error as string}
                      />
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        {current_status != "" && (
          <div className="w-full flex justify-start items-center gap-2">
            <FaCircle className="text-secondary text-sm pulsing" />
            <p className="text-sm shine">{current_status}</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default MessageDisplay;
