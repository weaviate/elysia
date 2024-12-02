"use client";

import React, { useEffect, useState } from "react";
import { handleNamedEntityRecognition } from "../api";
import { FaCopy } from "react-icons/fa";

interface UserMessageDisplayProps {
  payload: string[];
  onClick: () => void;
  collapsed: boolean;
}

const UserMessageDisplay: React.FC<UserMessageDisplayProps> = ({
  payload,
  onClick,
  collapsed,
}) => {
  const [nounSpans, setNounSpans] = useState<[number, number][]>([]);
  const [entitySpans, setEntitySpans] = useState<[number, number][]>([]);

  const text = payload && payload[0];

  useEffect(() => {
    handleNamedEntityRecognition(payload[0]).then((data) => {
      setNounSpans(data.noun_spans);
      setEntitySpans(data.entity_spans);
    });
  }, [payload]);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text || "");
  };

  const renderTextWithHighlights = (text: string) => {
    if (!text || (nounSpans.length === 0 && entitySpans.length === 0))
      return text;

    // Combine and sort spans
    const spans = [
      ...nounSpans.map(([start, end]) => ({ start, end, type: "noun" })),
      ...entitySpans.map(([start, end]) => ({ start, end, type: "entity" })),
    ];

    // Build events for span starts and ends
    const events: { index: number; type: string; isStart: boolean }[] = [];
    spans.forEach((span) => {
      events.push({ index: span.start, type: span.type, isStart: true });
      events.push({ index: span.end, type: span.type, isStart: false });
    });

    // Sort events by index
    events.sort((a, b) => a.index - b.index || (a.isStart ? -1 : 1));

    const segments: JSX.Element[] = [];
    let lastIndex = 0;
    const activeTypes = new Set<string>();

    events.forEach((event) => {
      if (event.index > lastIndex) {
        const segmentText = text.slice(lastIndex, event.index);
        let className = "";
        if (activeTypes.has("noun")) {
          className =
            "font-bold text-highlight [text-shadow:0_0_10px_theme(colors.highlightshine)] ";
        }
        if (activeTypes.has("entity")) {
          className =
            "text-accent font-bold [text-shadow:0_0_10px_theme(colors.accentshine)] ";
        }

        segments.push(
          <span
            key={`segment-${lastIndex}-${event.index}`}
            className={className}
          >
            {segmentText}
          </span>
        );
      }

      if (event.isStart) {
        activeTypes.add(event.type);
      } else {
        activeTypes.delete(event.type);
      }

      lastIndex = event.index;
    });

    // Add any remaining text after the last event
    if (lastIndex < text.length) {
      let className = "";

      if (activeTypes.has("noun")) {
        className =
          "font-bold text-highlight [text-shadow:0_0_10px_theme(colors.highlightshine)] ";
      }
      if (activeTypes.has("entity")) {
        className =
          "text-accent font-bold [text-shadow:0_0_10px_theme(colors.accentshine)] ";
      }

      segments.push(
        <span key={`segment-${lastIndex}-end`} className={className}>
          {text.slice(lastIndex)}
        </span>
      );
    }

    return segments;
  };

  return (
    <button
      className="flex flex-col rounded-lg transition-all duration-300 justify-start items-start mt-8 cursor-pointer w-full"
      onClick={onClick}
    >
      <div className="w-full">
        <div className="flex flex-grow justify-start items-start chat-animation gap-4">
          {!collapsed ? (
            <div className="flex gap-2 items-center">
              <p className="text-primary text-3xl text-left flex-grow">
                {renderTextWithHighlights(text)}
              </p>
              <button
                onClick={handleCopy}
                className="text-secondary hover:text-primary transition-all duration-300"
                title="Copy to clipboard"
              >
                <FaCopy size={12} />
              </button>
            </div>
          ) : (
            <p className="text-secondary hover:text-primary text-lg transition-all duration-300 text-left">
              {text}
            </p>
          )}
        </div>
      </div>
    </button>
  );
};

export default UserMessageDisplay;
