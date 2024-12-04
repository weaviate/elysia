"use client";

import MarkdownMessageDisplay from "./markdown";
import { useState } from "react";

interface EpicGenericContentProps {
  _text: string;
}

const EpicGenericContent: React.FC<EpicGenericContentProps> = ({ _text }) => {
  const max_length = 500;
  const text = _text ? _text : "";
  const [collapsed, setCollapsed] = useState(text && text.length > max_length);

  return (
    <div className="w-full flex flex-col gap-2 items-center justify-center">
      <MarkdownMessageDisplay
        text={collapsed ? text.slice(0, max_length) : text}
      />
      {text.length > max_length && (
        <>
          {collapsed ? (
            <button
              className="btn"
              onClick={() => setCollapsed((prev) => !prev)}
            >
              <p className="text-xs text-secondary">Show more</p>
            </button>
          ) : (
            <button
              className="btn"
              onClick={() => setCollapsed((prev) => !prev)}
            >
              <p className="text-xs text-secondary">Show less</p>
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default EpicGenericContent;
