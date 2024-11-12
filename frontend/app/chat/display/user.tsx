"use client";

import React, { useEffect, useState } from "react";

import { handleNamedEntityRecognition } from "../api";

interface UserMessageDisplayProps {
  payload: string[];
}

const UserMessageDisplay: React.FC<UserMessageDisplayProps> = ({ payload }) => {
  const [nounSpans, setNounSpans] = useState<[number, number][]>([]);

  const text = payload && payload[0];

  useEffect(() => {
    handleNamedEntityRecognition(payload[0]).then((data) => {
      setNounSpans(data.noun_spans);
    });
  }, []);

  const renderTextWithHighlights = (text: string) => {
    if (!text || nounSpans.length === 0) return text;

    const segments: JSX.Element[] = [];
    let lastIndex = 0;

    nounSpans.forEach(([start, end], i) => {
      // Add non-highlighted text before the span
      if (start > lastIndex) {
        segments.push(
          <span key={`text-${i}`}>{text.slice(lastIndex, start)}</span>
        );
      }
      // Add highlighted span
      segments.push(
        <span key={`highlight-${i}`} className="font-bold text-highlight">
          {text.slice(start, end)}
        </span>
      );
      lastIndex = end;
    });

    // Add remaining text after last span
    if (lastIndex < text.length) {
      segments.push(<span key="text-end">{text.slice(lastIndex)}</span>);
    }

    return segments;
  };

  return (
    <div className="w-full flex flex-col justify-start items-start mt-8 ">
      <div className="max-w-3/5">
        <div className="flex flex-grow justify-start items-start chat-animation">
          <p className="text-primary text-2xl font-bold">
            {renderTextWithHighlights(text)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default UserMessageDisplay;
