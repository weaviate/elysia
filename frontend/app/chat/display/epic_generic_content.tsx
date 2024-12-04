"use client";

import { EpicGeneric } from "@/app/types";
import MarkdownMessageDisplay from "./markdown";
import { useState } from "react";

interface EpicGenericContentProps {
  text: string;
}

const EpicGenericContent: React.FC<EpicGenericContentProps> = ({ text }) => {
  const [collapsed, setCollapsed] = useState(text.length > 500);

  return (
    <div className="w-full flex flex-col gap-2 items-center justify-center">
      <MarkdownMessageDisplay text={collapsed ? text.slice(0, 500) : text} />
      {collapsed ? (
        <button className="btn" onClick={() => setCollapsed((prev) => !prev)}>
          <p className="text-xs text-secondary">Show more</p>
        </button>
      ) : (
        <button className="btn" onClick={() => setCollapsed((prev) => !prev)}>
          <p className="text-xs text-secondary">Show less</p>
        </button>
      )}
    </div>
  );
};

export default EpicGenericContent;
