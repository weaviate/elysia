"use client";

import React from "react";

import { Message, TicketPayload } from "../types";

import UserMessageDisplay from "./user_message_display";
import MarkdownMessageDisplay from "./markdown_display";
import ErrorMessageDisplay from "./error_message_display";
import TicketMessageDisplay from "./ticket_display";
import { FaCircle } from "react-icons/fa6";

interface MessageDisplayProps {
  messages: Message[];
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({ messages }) => {
  return (
    <div className="h-[80vh] w-[60vw] flex justify-start items-start mt-10 ">
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
            {message.type === "Text" && (
              <div className="w-full flex flex-col justify-start items-start ">
                <div className="max-w-3/5">
                  <MarkdownMessageDisplay
                    key={index}
                    markdown_message={
                      typeof message.payload === "string" ? message.payload : ""
                    }
                  />
                </div>
              </div>
            )}
            {message.type === "Error" && (
              <div className="w-full flex flex-col justify-start items-start ">
                <div className="max-w-3/5">
                  <ErrorMessageDisplay
                    key={index}
                    error_message={
                      typeof message.payload === "string" ? message.payload : ""
                    }
                  />
                </div>
              </div>
            )}
            {message.type === "Ticket" && (
              <div className="w-full flex flex-col justify-start items-end ">
                <div className="max-w-3/5">
                  <TicketMessageDisplay
                    key={index}
                    ticket={message.payload as TicketPayload}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
        <div className="w-full flex justify-start items-center gap-2">
          <FaCircle className="text-secondary text-sm shine" />
          <p className="text-sm shine">Querying...</p>
        </div>
      </div>
    </div>
  );
};

export default MessageDisplay;
