"use client";

import React, { useEffect, useState } from "react";
import { handleNamedEntityRecognition } from "../api";

interface UserMessageDisplayProps {
  payload: string[];
}

const UserMessageDisplay: React.FC<UserMessageDisplayProps> = ({ payload }) => {
  const [nounSpans, setNounSpans] = useState<[number, number][]>([]);
  const [entitySpans, setEntitySpans] = useState<[number, number][]>([]);

  const text = payload && payload[0];

  useEffect(() => {
    handleNamedEntityRecognition(payload[0]).then((data) => {
      setNounSpans(data.noun_spans);
      setEntitySpans(data.entity_spans);
    });
  }, [payload]);

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
          className = "font-bold text-highlight ";
        }
        if (activeTypes.has("entity")) {
          className = "text-accent font-bold ";
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
        className = "font-bold text-highlight shine-highlight ";
      }
      if (activeTypes.has("entity")) {
        className = "text-accent font-bold shine-accent ";
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
    <div className="w-full flex flex-col justify-start items-start mt-8">
      <div className="max-w-3/5">
        <div className="flex flex-grow justify-start items-start chat-animation">
          <p className="text-primary text-2xl">
            {renderTextWithHighlights(text)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default UserMessageDisplay;
