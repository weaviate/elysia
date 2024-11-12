"use client";

import React from "react";

interface UserMessageDisplayProps {
  payload: string[];
}

const UserMessageDisplay: React.FC<UserMessageDisplayProps> = ({ payload }) => {
  const splitText = (text: string) => {
    const words = text.split(" ");
    const firstWord = words[0];
    const remainingWords = words.slice(1).join(" ");
    return { firstWord, remainingWords };
  };

  // TODO: Replace hardcoded text-highlight with API call

  return (
    <div className="w-full flex flex-col justify-start items-start mt-8 ">
      <div className="max-w-3/5">
        {payload.map((text, idx) => (
          <div
            key={idx}
            className="flex flex-grow justify-start items-start chat-animation"
          >
            <p className="text-primary text-2xl font-bold">
              <span className="text-highlight">
                {splitText(text).firstWord}
              </span>
              {splitText(text).remainingWords &&
                " " + splitText(text).remainingWords}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UserMessageDisplay;
