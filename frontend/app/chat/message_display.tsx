"use client";

import React from "react";

import { Message } from "../types";

import UserMessageDisplay from "./user_message_display";

interface MessageDisplayProps {
  messages: Message[];
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({ messages }) => {
  return (
    <div className="h-[80vh] w-[50vw] flex justify-start items-start mt-10 ">
      <div className="flex flex-col gap-2 w-full">
        {messages.map((message, index) => (
          <div key={index} className="w-full flex">
            <div className="w-full flex flex-col justify-end items-end ">
              <div className="w-3/5">
                <UserMessageDisplay
                  key={index}
                  user_message={message.user_message}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MessageDisplay;
