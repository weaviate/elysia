"use client";

import React, { useEffect, useState } from "react";
import { Collection, MetadataPayload } from "../types";
import { getCollectionMetadata } from "./api";
interface DashboardProps {
  conversation_id: string;
  user_id: string;
  collections: Collection[];
}

const Dashboard: React.FC<DashboardProps> = ({
  conversation_id,
  user_id,
  collections,
}) => {
  const [metadata, setMetadata] = useState<MetadataPayload | null>(null);

  const [selectedMetadata, setSelectedMetadata] = useState<string | null>(null);

  const fetchMetadata = async () => {
    const metadata = await getCollectionMetadata(conversation_id, user_id);
    setMetadata(metadata);
  };

  const selectMetadata = (key: string) => {
    if (selectedMetadata === key) {
      deselectMetadata();
    } else {
      setSelectedMetadata(key);
    }
  };

  const deselectMetadata = () => {
    setSelectedMetadata(null);
  };

  const collection_count = collections.length;
  const total_objects = collections.reduce(
    (acc, collection) => acc + collection.total,
    0
  );

  useEffect(() => {
    fetchMetadata();
  }, [user_id]);

  return (
    <div
      className="flex w-[80vw] overflow-y-auto gap-5 flex-col p-8 h-screen items-start justify-start outline-none"
      tabIndex={0}
    >
      <div className="flex flex-col gap-1 w-full">
        <p className="text-xl font-bold font-merriweather text-primary">
          Data Dashboard
        </p>
        <p className="text-secondary text-sm">
          {collection_count} collections / {total_objects} objects
        </p>
      </div>
      <div className="flex flex-wrap items-start justify-start gap-3">
        {metadata &&
          Object.keys(metadata.metadata || {}).map((key) => (
            <div
              key={key}
              onClick={() => selectMetadata(key)}
              className={`flex cursor-pointer flex-col gap-3 p-6 transition-all duration-300 hover:bg-foreground bg-background_alt rounded-lg ${
                selectedMetadata === key ? "w-[70vw]" : "w-[35vw]"
              }`}
            >
              <p className="text-primary text-lg font-merriweather font-bold">
                {metadata.metadata[key].name}
              </p>
              {selectedMetadata === key && (
                <div className="flex flex-col gap-6">
                  <div className="flex flex-row gap-4 justify-start items-center">
                    <div className="flex p-2 border border-primary rounded-lg">
                      <p className="text-primary text-sm">
                        {metadata.metadata[key].length} objects
                      </p>
                    </div>
                    <div className="flex p-2 border border-primary rounded-lg">
                      <p className="text-primary text-sm">
                        {Object.keys(metadata.metadata[key].fields).length}{" "}
                        fields
                      </p>
                    </div>
                    {Object.keys(metadata.metadata[key].mappings).map(
                      (mapping) => (
                        <div
                          key={mapping}
                          className="flex p-2 border border-primary rounded-lg"
                        >
                          <p className="text-primary text-sm font-bold">
                            {mapping}
                          </p>
                        </div>
                      )
                    )}
                  </div>
                  <p className="text-secondary text-sm">
                    {metadata.metadata[key].summary}
                  </p>
                  <hr className="border-t border-secondary" />
                  <div className="flex flex-wrap gap-6">
                    {Object.keys(metadata.metadata[key].fields).map((field) => (
                      <div key={field} className="flex flex-col gap-1">
                        <div className="flex gap-2 items-center justify-start">
                          <p className="text-primary text-sm font-bold">
                            {field}
                          </p>
                          <p className="text-secondary text-xs">
                            {metadata.metadata[key].fields[field].type}
                          </p>
                        </div>
                        <div className="grid grid-rows-3 auto-cols-max gap-2">
                          {metadata.metadata[key].fields[field].groups?.map(
                            (group, index) => (
                              <p key={index} className="text-secondary text-sm">
                                {group}
                              </p>
                            )
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  );
};

export default Dashboard;
