"use client";

import { DocumentPayload } from "@/app/types";
import EpicGenericContent from "./epic_generic_content";

interface DocumentDisplayProps {
  payload: DocumentPayload[];
}

const DocumentDisplay: React.FC<DocumentDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex overflow-x-scroll justify-start max-h-[50vh] p-4 items-start gap-3">
      {payload.map((p, index) => (
        <div
          key={`${p.uuid}-${index}`}
          className="p-8 bg-foreground shadow-xl rounded-lg w-[50vw] flex-shrink-0 flex flex-col gap-2"
        >
          <p className="text-sm font-medium">{p.title}</p>
          <EpicGenericContent _text={p.content} />
        </div>
      ))}
    </div>
  );
};

export default DocumentDisplay;
