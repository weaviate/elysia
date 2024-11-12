"use client";

import React from "react";
import { MdError } from "react-icons/md";
import MarkdownMessageDisplay from "./markdown";

interface ErrorMessageDisplayProps {
  error: string;
}

const ErrorMessageDisplay: React.FC<ErrorMessageDisplayProps> = ({ error }) => {
  return (
    <div className="w-full flex flex-col justify-start items-start ">
      <div className="max-w-3/5">
        <div className="flex flex-grow justify-start items-center gap-2 chat-animation bg-error p-4 rounded-lg">
          <MdError />
          <MarkdownMessageDisplay text={error} />
        </div>
      </div>
    </div>
  );
};

export default ErrorMessageDisplay;
