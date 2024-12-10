"use client";

import { DocumentPayload } from "@/app/types";
import ChunksDisplay from "./chunks_spans";
import { useState } from "react";
import { FaExpand, FaTrash } from "react-icons/fa";
import DocumentModal from "./document_modal";

interface DocumentDisplayProps {
  payload: DocumentPayload[];
}

const DocumentDisplay: React.FC<DocumentDisplayProps> = ({ payload }) => {
  const [currentDocument, setCurrentDocument] =
    useState<DocumentPayload | null>(null);

  const selectDocument = (document: DocumentPayload) => {
    setCurrentDocument(document);
  };

  const deselectDocument = () => {
    setCurrentDocument(null);
  };

  return (
    <div className="w-full flex overflow-x-scroll justify-start max-h-[50vh] p-4 items-start gap-3">
      {payload.map((p, index) => (
        <div
          key={`${p.uuid}-${index}`}
          className="p-8 bg-foreground cursor-pointer shadow-xl rounded-lg w-[50vw] flex-shrink-0 flex flex-col gap-2"
        >
          <div className="flex flex-row justify-between items-center w-full gap-2">
            <div className="flex flex-col gap-2">
              <p className="text-sm font-medium">{p.title}</p>
              <p className="text-xs text-primary">{p.author}</p>
            </div>
            <div className="flex">
              <button
                onClick={() => selectDocument(p)}
                className="flex flex-row items-center transition-all duration-300 gap-2 text-secondary hover:text-primary"
              >
                <p className="text-xs ">Show Source</p>
                <FaExpand />
              </button>
            </div>
          </div>
          <ChunksDisplay _text={p.content} chunk_spans={p.chunk_spans} />
        </div>
      ))}
      {currentDocument && (
        <DocumentModal document={currentDocument} onClose={deselectDocument} />
      )}
    </div>
  );
};

export default DocumentDisplay;
