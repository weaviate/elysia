"use client";

import React from "react";

import { Message } from "../types";

interface UserMessageDisplayProps {
  user_message: string;
}

const UserMessageDisplay: React.FC<UserMessageDisplayProps> = ({
  user_message,
}) => {
  const words = user_message.split(" ");
  const firstWord = words[0];
  const remainingWords = words.slice(1).join(" ");

  return (
    <div className="flex flex-grow justify-start items-start chat-animation">
      <p className="text-primary text-2xl font-bold">
        <span className="text-highlight">{firstWord}</span>
        {remainingWords && " " + remainingWords}
      </p>
    </div>
  );
};

export default UserMessageDisplay;
