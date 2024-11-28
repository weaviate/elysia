"use client";

import React from "react";

interface DataTableProps {
  data: { [key: string]: any }[] | null;
  header: string[];
  loading: boolean;
  setSelectedCell: (cell: { [key: string]: any }) => void;
}

const DataTable: React.FC<DataTableProps> = ({
  data,
  header,
  loading,
  setSelectedCell,
}) => {
  if (!data) return null;

  return (
    <div className="h-[80vh] items-center overflow-scroll w-full flex flex-col p-8 mt-4">
      {loading && <p className="text-primary shine">Loading...</p>}
      {!loading && data && (
        <div className="w-full flex-col gap-2 items-center justify-center">
          <table className="table table-auto w-full fade-in">
            <thead className="table-header-group">
              <tr className="">
                {header.map((key) => (
                  <th
                    key={key}
                    className="p-2 cursor-pointer items-start justify-start bg-background  border-2 border-background rounded-md"
                  >
                    <p className="truncate text-xs text-primary font-light">
                      {key}
                    </p>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((item, index) => (
                <tr
                  key={index}
                  className="bg-background hover:text-primary text-secondary hover:bg-foreground transition-all cursor-pointer duration-300 ease-in-out"
                  onClick={() => setSelectedCell(item)}
                >
                  {header.map((key) => (
                    <td
                      key={key}
                      className="p-3 whitespace-nowrap min-w-[100px] max-w-[400px] truncate"
                    >
                      <div className="flex items-center justify-start">
                        <p className="text-xs font-light">
                          {item[key] !== undefined ? item[key] : ""}
                        </p>
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default DataTable;
