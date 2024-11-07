"use client";

import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { MdError } from "react-icons/md";

interface ErrorMessageDisplayProps {
  error_message: string;
}

const ErrorMessageDisplay: React.FC<ErrorMessageDisplayProps> = ({
  error_message,
}) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let currentIndex = 0;
    const messageLength = error_message.length;
    const interval = 30; // Adjust typing speed (milliseconds per character)
    let timeoutId: NodeJS.Timeout;

    const typeWriter = () => {
      if (currentIndex <= messageLength) {
        setDisplayedText(error_message.slice(0, currentIndex));
        currentIndex++;
        timeoutId = setTimeout(typeWriter, interval);
      }
    };

    typeWriter();

    return () => {
      clearTimeout(timeoutId);
    };
  }, [error_message]);

  return (
    <div className="flex flex-grow justify-start items-center gap-2 chat-animation text-error ">
      <MdError className="" />
      <ReactMarkdown>{displayedText}</ReactMarkdown>
    </div>
  );
};

export default ErrorMessageDisplay;
