"use client";

import React, { useEffect, useState } from "react";

import {
  ConversationMessage,
  Message,
  ResponsePayload,
  SummaryPayload,
  ResultPayload,
  TextPayload,
  AggregationPayload,
  EpicGeneric,
} from "../../types";

import UserMessageDisplay from "./user";
import ErrorMessageDisplay from "./error";
import TextDisplay from "./text";
import TicketsDisplay from "./tickets";
import WarningDisplay from "./warning";
import ConversationsDisplay from "./conversations";
import SummaryDisplay from "./summary";
import EcommerceDisplay from "./ecommerce";
import BoringGenericDisplay from "./boring_generic";
import CollectionDisplay from "./collection";
import CodeDisplay from "./code";
import AggregationDisplay from "./aggregation";
import EpicGenericDisplay from "./epic_generic";

interface MessageDisplayProps {
  messages: Message[];
  _collapsed: boolean;
  routerChangeCollection: (collection_id: string) => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({
  messages,
  routerChangeCollection,
  _collapsed,
  messagesEndRef,
}) => {
  const [displayMessages, setDisplayMessages] = useState<Message[]>([]);

  const [collapsed, setCollapsed] = useState<boolean>(_collapsed);

  const mergeMessages = (messages: Message[]) => {
    const newMessages: Message[] = [];
    const skip_indices: number[] = [];

    messages.forEach((message, index) => {
      if (skip_indices.includes(index)) {
        return;
      } else if (
        message.type === "text" &&
        (message.payload as ResponsePayload).type === "response" &&
        !(index + 1 == messages.length)
      ) {
        const content: TextPayload[] = [
          (message.payload as ResponsePayload).objects[0] as TextPayload,
        ];
        const next_message_id: string = message.id;

        for (let i = index + 1; i < messages.length; i++) {
          if (
            messages[i].type === "text" &&
            (messages[i].payload as ResponsePayload).type === "response"
          ) {
            content.push(
              (messages[i].payload as ResponsePayload).objects[0] as TextPayload
            );
            skip_indices.push(i);
          } else {
            break;
          }
        }

        const newResponsePayload: ResponsePayload = {
          type: "response",
          metadata: (message.payload as ResponsePayload).metadata,
          objects: content,
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
    console.log(messages);
  }, [messages]);

  return (
    <div
      className={`flex justify-start items-start p-4 transition-all duration-300`}
    >
      <div className="flex flex-col gap-6 w-full">
        {displayMessages
          .filter((m) => m.type === "User")
          .map((message, index) => (
            <div key={`${index}-${message.id}-message`} className="w-full flex">
              {message.type === "User" && (
                <UserMessageDisplay
                  onClick={() => setCollapsed((prev) => !prev)}
                  key={`${index}-${message.id}-user`}
                  payload={
                    (message.payload as ResultPayload).objects as string[]
                  }
                  collapsed={collapsed}
                />
              )}
            </div>
          ))}
        {!collapsed && (
          <div className="flex flex-col gap-6">
            {displayMessages
              .filter((m) => m.type !== "User")
              .map((message, index) => (
                <div
                  key={`${index}-${message.id}-message`}
                  className="w-full flex"
                >
                  {message.type === "result" && (
                    <div className="w-full flex flex-col justify-start items-start gap-3">
                      <div className="flex flex-col gap-3 w-full">
                        {(message.payload as ResultPayload).metadata[
                          "collection_name"
                        ] && (
                          <CollectionDisplay
                            collection_name={
                              (message.payload as ResultPayload).metadata[
                                "collection_name"
                              ]
                            }
                            total_objects={
                              (message.payload as ResultPayload).objects.length
                            }
                            routerChangeCollection={routerChangeCollection}
                          />
                        )}
                        {(message.payload as ResultPayload).code && (
                          <CodeDisplay
                            payload={(message.payload as ResultPayload).code}
                          />
                        )}
                      </div>
                      {(message.payload as ResultPayload).type === "ticket" && (
                        <TicketsDisplay
                          key={`${index}-${message.id}-tickets`}
                          message={message}
                        />
                      )}
                      {(message.payload as ResultPayload).type ===
                        "ecommerce" && (
                        <EcommerceDisplay
                          key={`${index}-${message.id}-ecommerce`}
                          payload={message.payload as ResultPayload}
                        />
                      )}
                      {((message.payload as ResultPayload).type ===
                        "conversation" ||
                        (message.payload as ResultPayload).type ===
                          "message") && (
                        <ConversationsDisplay
                          key={`${index}-${message.id}-conversations`}
                          payload={
                            (message.payload as ResultPayload).type ===
                            "conversation"
                              ? ((message.payload as ResultPayload)
                                  .objects as ConversationMessage[][])
                              : [
                                  (message.payload as ResultPayload)
                                    .objects as ConversationMessage[],
                                ]
                          }
                        />
                      )}
                      {((message.payload as ResultPayload).type ===
                        "boring_generic" ||
                        (message.payload as ResultPayload).type ===
                          "mapped") && (
                        <>
                          <p className="text-sm text-secondary">
                            {(message.payload as ResultPayload).type}
                          </p>
                          <BoringGenericDisplay
                            key={`${index}-${message.id}-boring_generic`}
                            payload={
                              (message.payload as ResultPayload).objects as {
                                [key: string]: string;
                              }[]
                            }
                          />
                        </>
                      )}
                      {(message.payload as ResultPayload).type ===
                        "epic_generic" && (
                        <EpicGenericDisplay
                          key={`${index}-${message.id}-epic_generic`}
                          payload={
                            (message.payload as ResultPayload)
                              .objects as EpicGeneric[]
                          }
                        />
                      )}
                      {(message.payload as ResultPayload).type ===
                        "aggregation" && (
                        <AggregationDisplay
                          key={`${index}-${message.id}-aggregation`}
                          aggregation={
                            (message.payload as ResultPayload)
                              .objects as AggregationPayload[]
                          }
                        />
                      )}
                    </div>
                  )}
                  {message.type === "text" && (
                    <div className="w-full flex flex-col justify-start items-start ">
                      {(message.payload as ResponsePayload).type ===
                        "response" && (
                        <TextDisplay
                          key={`${index}-${message.id}-response`}
                          payload={
                            (message.payload as ResponsePayload)
                              .objects as TextPayload[]
                          }
                        />
                      )}
                      {(message.payload as ResponsePayload).type ===
                        "summary" && (
                        <SummaryDisplay
                          key={`${index}-${message.id}-summary`}
                          payload={
                            (message.payload as ResponsePayload)
                              .objects as SummaryPayload[]
                          }
                        />
                      )}
                    </div>
                  )}
                  {message.type === "error" && (
                    <ErrorMessageDisplay
                      key={`${index}-${message.id}-error`}
                      error={(message.payload as TextPayload).text}
                    />
                  )}
                  {message.type === "warning" && (
                    <WarningDisplay
                      key={`${index}-${message.id}-warning`}
                      warning={(message.payload as TextPayload).text}
                    />
                  )}
                </div>
              ))}
          </div>
        )}
        {!collapsed && <div ref={messagesEndRef} />}
      </div>
    </div>
  );
};

export default MessageDisplay;
