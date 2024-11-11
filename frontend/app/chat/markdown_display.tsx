"use client";

import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import Typewriter from "typewriter-effect";

interface MarkdownMessageDisplayProps {
  markdown_message: string;
}

const MarkdownMessageDisplay: React.FC<MarkdownMessageDisplayProps> = ({
  markdown_message,
}) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let currentIndex = 0;
    const messageLength = markdown_message.length;
    const interval = 4; // Adjust typing speed (milliseconds per character)
    let timeoutId: NodeJS.Timeout;

    const typeWriter = () => {
      if (currentIndex <= messageLength) {
        setDisplayedText(markdown_message.slice(0, currentIndex));
        currentIndex++;
        timeoutId = setTimeout(typeWriter, interval);
      }
    };

    typeWriter();

    return () => {
      clearTimeout(timeoutId);
    };
  }, [markdown_message]);

  return (
    <div className="flex gap-8 flex-col flex-grow justify-start text-sm items-start chat-animation text-white text-wrap">
      <ReactMarkdown>{markdown_message}</ReactMarkdown>
    </div>
  );
};

export default MarkdownMessageDisplay;
