"use client";

import { CodePayload } from "@/app/types";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { FaCode } from "react-icons/fa";
import { useState } from "react";
import { FaCopy } from "react-icons/fa";

interface CodeDisplayProps {
  payload: CodePayload;
}

const CodeDisplay: React.FC<CodeDisplayProps> = ({ payload }) => {
  const [collapsed, setCollapsed] = useState<boolean>(true);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="w-full flex flex-col justify-start items-start mb-4">
      <div className="text-sm chat-animation text-white w-full">
        <div
          onClick={() => setCollapsed((prev) => !prev)}
          className="flex items-center justify-between gap-2 p-3 border hover:text-accent hover:border-accent transition-all duration-300 text-secondary border-secondary rounded-lg w-full cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <FaCode className="" />
            <p className="font-bold text-xs">{payload.title}</p>
          </div>
          <button
            className="flex items-center gap-2 btn btn-round"
            onClick={(e) => {
              e.stopPropagation();
              copyToClipboard(payload.text);
            }}
          >
            <FaCopy size={12} />
          </button>
        </div>
        {!collapsed && (
          <SyntaxHighlighter
            language={payload.language}
            wrapLongLines={true}
            showLineNumbers={true}
            style={oneDark}
            customStyle={{ backgroundColor: "#301B29", color: "#ffffff" }}
            className="rounded-lg p-5 w-full text-sm shadow-xl"
          >
            {payload.text}
          </SyntaxHighlighter>
        )}
      </div>
    </div>
  );
};

export default CodeDisplay;
