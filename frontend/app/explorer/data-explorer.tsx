"use client";

import React, { useEffect, useState } from "react";

import { Collection, CollectionData } from "../types";
import { IoMdCloseCircle } from "react-icons/io";
import ReactMarkdown from "react-markdown";

import { TbPlayerTrackNextFilled } from "react-icons/tb";
import { TbPlayerTrackPrevFilled } from "react-icons/tb";
import { TbPlayerSkipForwardFilled } from "react-icons/tb";
import { TbPlayerSkipBackFilled } from "react-icons/tb";

import DataTable from "./table";

interface DataExplorerProps {
  collectionData: CollectionData | null;
  collectionLoading: boolean;
  collectionName: string;
  collection: Collection | null;
  pageDownMax: () => void;
  pageDown: () => void;
  pageUpMax: () => void;
  pageUp: () => void;
  maxPage: number;
  pageSize: number;
  page: number;
}

const DataExplorer: React.FC<DataExplorerProps> = ({
  collectionData,
  collectionLoading,
  collectionName,
  collection,
  pageDownMax,
  pageDown,
  pageUpMax,
  pageUp,
  maxPage,
  pageSize,
  page,
}) => {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const [selectedCell, setSelectedCell] = useState<{
    [key: string]: any;
  } | null>(null);

  useEffect(() => {
    setSelectedCell(null);
  }, [collectionName]);

  if (!collection) return null;

  return (
    <div
      className="flex w-[80vw] flex-col p-4 h-screen items-start justify-start outline-none"
      onKeyDown={(e) => e.key === "Escape" && setSelectedCell(null)}
      tabIndex={0}
    >
      <div className="flex flex-col gap-1 items-start justify-start w-full">
        <p className="text-primary text-xl font-bold">{collectionName}</p>
        <div className="flex items-center justify-start gap-1">
          <p className="text-secondary text-xs font-light">
            {collection.total} items
          </p>
          <p className="text-secondary text-xs font-light">
            {collection.vectorizer}
          </p>
        </div>
      </div>
      {!selectedCell ? (
        <DataTable
          data={collectionData?.items || []}
          header={Object.keys(collectionData?.properties || {})}
          loading={collectionLoading}
          setSelectedCell={setSelectedCell}
        />
      ) : (
        <div className="h-[90vh] items-start justify-start gap-5 overflow-scroll w-full flex p-4 flex-col mt-4 fade-in">
          <div className="flex items-end justify-end w-full">
            <button
              className="btn text-sm text-primary"
              onClick={() => setSelectedCell(null)}
            >
              <p>Close</p>
              <IoMdCloseCircle className="text-primary" />
            </button>
          </div>

          <div className="flex flex-wrap gap-4 w-full">
            {Object.keys(selectedCell).map((key) => (
              <div
                className="flex flex-col flex-grow gap-1 bg-background_alt overflow-scroll p-5 rounded-lg text-wrap break-words max-h-[40vh] min-w-[48%] max-w-[100%]"
                key={key}
              >
                <p className="text-secondary text-xs font-light">{key}</p>
                <div className="text-sm text-primary whitespace-pre-wrap">
                  <ReactMarkdown>{String(selectedCell[key])}</ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!selectedCell && (
        <div className="flex items-center justify-between w-full mt-5">
          <div className="flex-1"></div>
          <div className="flex items-center justify-center gap-2">
            <button
              className="btn btn-round text-primary"
              onClick={() => pageDownMax()}
            >
              <TbPlayerTrackPrevFilled />
            </button>
            <button
              className="btn btn-round text-primary"
              onClick={() => pageDown()}
            >
              <TbPlayerSkipBackFilled />
            </button>
            <p className="text-primary text-xs font-light">
              {"Page " + (page + 1) + " of " + (maxPage + 1)}
            </p>
            <button
              className="btn btn-round text-primary"
              onClick={() => pageUp()}
            >
              <TbPlayerTrackNextFilled />
            </button>
            <button
              className="btn btn-round text-primary"
              onClick={() => pageUpMax()}
            >
              <TbPlayerSkipForwardFilled />
            </button>
          </div>
          <div className="flex-1 flex items-center justify-end gap-2">
            <p className="text-secondary text-xs font-light">
              {"Page Size: " + pageSize}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataExplorer;
