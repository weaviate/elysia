"use client";

import React from "react";

import { Message } from "../types";

interface UserMessageDisplayProps {
  user_message: string;
}

const UserMessageDisplay: React.FC<UserMessageDisplayProps> = ({
  user_message,
}) => {
  return (
    <div className="flex flex-grow justify-start items-start chat-animation">
      <p className="text-primary text-2xl">{user_message}</p>
    </div>
  );
};

export default UserMessageDisplay;
