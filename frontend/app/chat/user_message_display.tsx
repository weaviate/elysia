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
    <div className="bg-foreground px-8 py-5 rounded-xl w-full flex justify-start items-start">
      <p className="text-primary text-sm">{user_message}</p>
    </div>
  );
};

export default UserMessageDisplay;
