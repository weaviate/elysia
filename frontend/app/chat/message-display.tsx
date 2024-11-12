"use client";

import React, { useEffect, useRef } from "react";

import {
  ConversationMessage,
  ErrorPayload,
  Message,
  ResultPayload,
} from "../types";

import UserMessageDisplay from "./display/user";
import ErrorMessageDisplay from "./display/error";
import TextDisplay from "./display/text";
import { FaCircle } from "react-icons/fa6";
import TicketsDisplay from "./display/tickets";
import WarningDisplay from "./display/warning";
import ConversationsDisplay from "./display/conversations";
import ConversationMessageDisplay from "./display/conversation-message";

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
  const size_control =
    messages.length == 0 ? "h-[0px] pb-0" : "h-[100vh] pb-32";

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, current_status]);

  return (
    <div
      className={`w-[75vw] flex justify-start items-start p-4 overflow-scroll transition-all duration-300 ${size_control}`}
    >
      <div className="flex flex-col gap-6 w-full">
        {messages.map((message, index) => (
          <div key={index + "message"} className="w-full flex">
            {message.type === "User" && (
              <UserMessageDisplay
                key={`${index}-${message.id}`}
                payload={(message.payload as ResultPayload).objects as string[]}
              />
            )}
            {message.type === "result" && (
              <div className="w-full flex flex-col justify-start items-start ">
                {(message.payload as ResultPayload).type === "text" && (
                  <TextDisplay
                    key={`${index}-${message.id}`}
                    payload={
                      (message.payload as ResultPayload).objects as string[]
                    }
                  />
                )}
                {(message.payload as ResultPayload).type === "ticket" && (
                  <TicketsDisplay
                    key={`${index}-${message.id}`}
                    message={message}
                  />
                )}
                {(message.payload as ResultPayload).type === "message" && (
                  <ConversationMessageDisplay
                    key={`${index}-${message.id}`}
                    payload={
                      (message.payload as ResultPayload)
                        .objects as ConversationMessage[]
                    }
                  />
                )}
                {(message.payload as ResultPayload).type === "conversation" && (
                  <ConversationsDisplay
                    key={`${index}-${message.id}`}
                    payload={
                      (message.payload as ResultPayload)
                        .objects as ConversationMessage[][]
                    }
                  />
                )}
              </div>
            )}
            {message.type === "error" && (
              <ErrorMessageDisplay
                key={`${index}-${message.id}`}
                error={(message.payload as ErrorPayload).error}
              />
            )}
            {message.type === "warning" && (
              <WarningDisplay
                key={`${index}-${message.id}`}
                payload={(message.payload as ResultPayload).objects as string[]}
              />
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
