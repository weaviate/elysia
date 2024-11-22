"use client";

import React from "react";

import { FaDatabase } from "react-icons/fa";

interface CollectionDisplayProps {
  collection_name: string | null;
  total_objects: number;
  routerChangeCollection: (collection_id: string) => void;
}

const CollectionDisplay: React.FC<CollectionDisplayProps> = ({
  collection_name,
  total_objects,
  routerChangeCollection,
}) => {
  return (
    collection_name && (
      <div
        onClick={(e) => {
          e.stopPropagation();
          routerChangeCollection(collection_name);
        }}
        className="flex justify-between items-center transition-all duration-300 cursor-pointer hover:text-primary text-secondary rounded-lg"
      >
        <div className="flex items-center gap-2">
          <FaDatabase className="" />
          <p className="text-xs font-bold ">
            {collection_name}
            {" ("}
            {total_objects}
            {") "}
          </p>
        </div>
      </div>
    )
  );
};

export default CollectionDisplay;
