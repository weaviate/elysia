"use client";

import { DocumentPayload } from "@/app/types";
import ChunksDisplay from "./chunks_spans";
import { useState } from "react";
import { IoMdCloseCircle } from "react-icons/io";
import MarkdownMessageDisplay from "./markdown";

interface DocumentModalProps {
  document: DocumentPayload;
  onClose: () => void;
}

const DocumentModal: React.FC<DocumentModalProps> = ({ document, onClose }) => {
  const renderContent = () => {
    if (!document.chunk_spans || document.chunk_spans.length === 0) {
      return <MarkdownMessageDisplay text={document.content} />;
    }

    const chunks: JSX.Element[] = [];
    let lastIndex = 0;

    document.chunk_spans.forEach(([start, end], index) => {
      // Add non-chunk text before the chunk
      if (lastIndex < start) {
        chunks.push(
          <MarkdownMessageDisplay
            key={`normal-${index}`}
            text={document.content.slice(lastIndex, start)}
          />
        );
      }

      // Add chunk text with highlighting
      chunks.push(
        <div className="flex flex-col gap-2">
          <div className="relative">
            <span
              key={`chunk-${index}`}
              className="italic font-bold text-primary drop-shadow-[0_0_0.3px_#ffffff] border border-primary rounded-md p-3 border-dashed block"
            >
              {document.content.slice(start, end)}
            </span>
            <div className="absolute -top-3.5 left-1/2 transform -translate-x-1/2 bg-background px-2">
              <span className="text-sm text-primary">Chunk {index + 1}</span>
            </div>
          </div>
        </div>
      );

      lastIndex = end;
    });

    // Add remaining text after last chunk
    if (lastIndex < document.content.length) {
      chunks.push(
        <MarkdownMessageDisplay
          key="normal-last"
          text={document.content.slice(lastIndex)}
        />
      );
    }

    return chunks;
  };

  return (
    <div className="fixed inset-0 fade-in bg-black bg-opacity-50 z-50 flex items-center justify-center">
      <div className="bg-background rounded-lg w-[70vw] max-w-4xl max-h-[90vh] overflow-hidden flex flex-col p-10">
        <div className="p-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <p className="text-xl text-primary">{document.title}</p>
            <p className="text-sm text-secondary">{document.author}</p>
          </div>
          <button
            onClick={onClose}
            className="text-secondary hover:text-primary transition-all duration-300"
            aria-label="Close modal"
          >
            <IoMdCloseCircle size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-6">
          <div className="flex flex-col gap-4">{renderContent()}</div>
        </div>
      </div>
    </div>
  );
};

export default DocumentModal;
