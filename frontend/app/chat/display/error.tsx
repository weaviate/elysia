"use client";

import React from "react";
import { MdError } from "react-icons/md";
import MarkdownMessageDisplay from "./markdown";

interface ErrorMessageDisplayProps {
  payload: string[];
}

const ErrorMessageDisplay: React.FC<ErrorMessageDisplayProps> = ({
  payload,
}) => {
  return (
    <div className="w-full flex flex-col justify-start items-start ">
      <div className="max-w-3/5">
        {payload.map((error, idx) => (
          <div
            key={`${idx}-${error}`}
            className="flex flex-grow justify-start items-center gap-2 chat-animation text-error "
          >
            <MdError />
            <MarkdownMessageDisplay text={error} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default ErrorMessageDisplay;
