"use client";

import React, { useEffect, useState } from "react";

import { Collection, CollectionData } from "../types";
import { IoMdCloseCircle } from "react-icons/io";
import ReactMarkdown from "react-markdown";

interface DataExplorerProps {
  collectionData: CollectionData | null;
  collectionLoading: boolean;
  collectionName: string;
  collection: Collection;
}

const DataExplorer: React.FC<DataExplorerProps> = ({
  collectionData,
  collectionLoading,
  collectionName,
  collection,
}) => {
  const [selectedCell, setSelectedCell] = useState<{
    [key: string]: any;
  } | null>(null);

  useEffect(() => {
    setSelectedCell(null);
  }, [collectionName]);

  return (
    <div
      className="flex flex-col w-full h-screen items-center justify-center outline-none"
      onKeyDown={(e) => e.key === "Escape" && setSelectedCell(null)}
      tabIndex={0}
    >
      <div className="flex flex-col gap-1 items-start justify-start w-[80vw]">
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
        <div className="h-[90vh] items-center overflow-scroll w-[80vw] flex flex-col p-8 mt-4">
          {collectionLoading && (
            <p className="text-primary shine">Loading...</p>
          )}
          {!collectionLoading && collectionData && (
            <table className="table-auto w-full fade-in">
              <thead>
                <tr>
                  {Object.keys(collectionData.properties).map((key) => (
                    <th
                      key={key}
                      className="p-2 items-start justify-start bg-background_alt text-secondary border-2 border-background rounded-md"
                    >
                      <p className="truncate text-sm text-secondary font-light">
                        {key}
                      </p>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {collectionData.items.map((item, index) => (
                  <tr
                    key={index}
                    className="hover:scale-105 bg-background_alt hover:bg-foreground transition-all cursor-pointer duration-300 ease-in-out"
                    onClick={() => setSelectedCell(item)}
                  >
                    {Object.keys(collectionData.properties).map((key) => (
                      <td
                        key={key}
                        className="p-3  border text-primary  border-background"
                      >
                        <div className="w-[150px] text-sm text-primary font-light truncate">
                          {item[key] !== undefined ? item[key] : ""}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ) : (
        <div className="h-[90vh] items-start justify-start gap-5 overflow-scroll w-[80vw] flex p-4 flex-col mt-4 fade-in">
          <div className="flex items-end justify-end w-full">
            <button
              className="btn text-primary"
              onClick={() => setSelectedCell(null)}
            >
              <p>Close</p>
              <IoMdCloseCircle className="text-primary" />
            </button>
          </div>

          {Object.keys(selectedCell).map((key) => (
            <div className="flex flex-col gap-1" key={key}>
              <p className="text-secondary text-xs font-light">{key}</p>
              <div className="text-base text-primary">
                <ReactMarkdown>{String(selectedCell[key])}</ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DataExplorer;
