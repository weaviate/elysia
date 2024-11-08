"use client";

import React from "react";

import { Message, ResultPayload } from "../types";

import UserMessageDisplay from "./user_message_display";
import MarkdownMessageDisplay from "./markdown_display";
import ErrorMessageDisplay from "./error_message_display";
import TicketMessageDisplay from "./ticket_display";
import { FaCircle } from "react-icons/fa6";

interface MessageDisplayProps {
  messages: Message[];
  current_status: string;
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({
  messages,
  current_status,
}) => {
  return (
    <div className="h-[80vh] w-[60vw] flex justify-start items-start mt-10 p-4 overflow-scroll ">
      <div className="flex flex-col gap-10 w-full">
        {messages.map((message, index) => (
          <div key={index + "message"} className="w-full flex">
            {message.type === "User" && (
              <div className="w-full flex flex-col justify-start items-start ">
                <div className="max-w-3/5">
                  <UserMessageDisplay
                    key={index}
                    user_message={
                      typeof message.payload === "string" ? message.payload : ""
                    }
                  />
                </div>
              </div>
            )}
            {message.type === "result" && (
              <div className="w-full flex flex-col justify-start items-start ">
                <div className="max-w-3/5">
                  {(message.payload as ResultPayload).type === "text" &&
                    (message.payload as ResultPayload).objects.map(
                      (text, idx) => (
                        <MarkdownMessageDisplay
                          key={`${index}-${idx}`}
                          markdown_message={text as string}
                        />
                      )
                    )}
                  {(message.payload as ResultPayload).type === "ticket" &&
                    (message.payload as ResultPayload).objects.map(
                      (ticket, idx) => (
                        <TicketMessageDisplay
                          key={`${index}-${idx}`}
                          ticket={ticket}
                        />
                      )
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
      </div>
    </div>
  );
};

export default MessageDisplay;
