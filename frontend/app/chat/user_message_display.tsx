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
    <div className="bg-foreground p-4 rounded-lg">
      <p className="font-semibold text-primary text-sm">{user_message}</p>
    </div>
  );
};

export default UserMessageDisplay;
