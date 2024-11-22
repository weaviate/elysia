"use client";

import React, { useEffect, useRef, useState } from "react";

import {
  ConversationMessage,
  ErrorPayload,
  Message,
  ResponsePayload,
  SummaryPayload,
  ResultPayload,
  TextPayload,
  CodePayload,
} from "../types";

import UserMessageDisplay from "./display/user";
import ErrorMessageDisplay from "./display/error";
import TextDisplay from "./display/text";
import { FaCircle } from "react-icons/fa6";
import TicketsDisplay from "./display/tickets";
import WarningDisplay from "./display/warning";
import ConversationsDisplay from "./display/conversations";
import ConversationMessageDisplay from "./display/conversation-message";
import SummaryDisplay from "./display/summary";
import CodeDisplay from "./display/code";

interface MessageDisplayProps {
  messages: Message[];
  current_status: string;
  routerChangeCollection: (collection_id: string) => void;
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({
  messages,
  current_status,
  routerChangeCollection,
}) => {
  const size_control =
    messages.length == 0 ? "h-[0px] pb-0" : "h-[100vh] pb-32";

  const [displayMessages, setDisplayMessages] = useState<Message[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const mergeMessages = (messages: Message[]) => {
    let newMessages: Message[] = [];
    let skip_indices: number[] = [];

    messages.forEach((message, index) => {
      if (skip_indices.includes(index)) {
        return;
      } else if (
        message.type === "text" &&
        (message.payload as ResponsePayload).type === "response" &&
        !(index + 1 == messages.length)
      ) {
        let content: string = (message.payload as ResponsePayload).objects[0]
          .text;
        let next_message_id: string = message.id;

        for (let i = index + 1; i < messages.length; i++) {
          if (
            messages[i].type === "text" &&
            (messages[i].payload as ResponsePayload).type === "response"
          ) {
            content +=
              " " +
              (
                (messages[i].payload as ResponsePayload)
                  .objects as TextPayload[]
              )[0].text;
            skip_indices.push(i);
            next_message_id = messages[i].id;
          } else {
            break;
          }
        }

        const newResponsePayload: ResponsePayload = {
          type: "response",
          metadata: (message.payload as ResponsePayload).metadata,
          objects: [{ text: content }],
        };

        const newMessage: Message = {
          ...message,
          payload: newResponsePayload,
          id: next_message_id,
        };

        newMessages.push(newMessage);
      } else {
        // For any other message type, just add it
        newMessages.push(message);
      }
    });

    setDisplayMessages(newMessages);
  };

  useEffect(() => {
    mergeMessages(messages);
  }, [messages]);

  useEffect(() => {
    scrollToBottom();
  }, [displayMessages, current_status]);

  return (
    <div
      className={`w-[75vw] flex justify-start items-start p-4 overflow-scroll transition-all duration-300 ${size_control}`}
    >
      <div className="flex flex-col gap-12 w-full">
        {displayMessages.map((message, index) => (
          <div key={index + "message"} className="w-full flex">
            {message.type === "User" && (
              <UserMessageDisplay
                key={`${index}-${message.id}`}
                payload={(message.payload as ResultPayload).objects as string[]}
              />
            )}
            {message.type === "result" && (
              <div className="w-full flex flex-col justify-start items-start ">
                {(message.payload as ResultPayload).type === "ticket" && (
                  <TicketsDisplay
                    key={`${index}-${message.id}`}
                    message={message}
                    routerChangeCollection={routerChangeCollection}
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
                    metadata={(message.payload as ResultPayload).metadata}
                    payload={
                      (message.payload as ResultPayload)
                        .objects as ConversationMessage[][]
                    }
                    routerChangeCollection={routerChangeCollection}
                  />
                )}
              </div>
            )}
            {message.type === "text" && (
              <div className="w-full flex flex-col justify-start items-start ">
                {(message.payload as ResponsePayload).type === "response" && (
                  <TextDisplay
                    key={`${index}-${message.id}`}
                    payload={
                      (message.payload as ResponsePayload)
                        .objects as TextPayload[]
                    }
                  />
                )}
                {(message.payload as ResponsePayload).type === "summary" && (
                  <SummaryDisplay
                    key={`${index}-${message.id}`}
                    payload={
                      (message.payload as ResponsePayload)
                        .objects as SummaryPayload[]
                    }
                  />
                )}
                {(message.payload as ResponsePayload).type === "code" && (
                  <CodeDisplay
                    key={`${index}-${message.id}`}
                    payload={
                      (message.payload as ResponsePayload)
                        .objects as CodePayload[]
                    }
                  />
                )}
              </div>
            )}
            {message.type === "error" && (
              <ErrorMessageDisplay
                key={`${index}-${message.id}`}
                error={(message.payload as TextPayload).text}
              />
            )}
            {message.type === "warning" && (
              <WarningDisplay
                key={`${index}-${message.id}`}
                warning={(message.payload as TextPayload).text}
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
