"use client";

import React, { useState } from "react";
import {
  AggregationCollection,
  AggregationField,
  AggregationPayload,
} from "@/app/types";
import KPIDisplay from "./kpi";

interface AggregationFieldDisplayProps {
  fieldName: string;
  field: AggregationField;
  depth: number;
}

const AggregationFieldDisplay: React.FC<AggregationFieldDisplayProps> = ({
  fieldName,
  field,
  depth,
}) => {
  const { type, values, groups } = field;
  return (
    <div className="flex flex-col gap-6 w-full  ">
      <KPIDisplay parent_field={fieldName} values={values} type={type} />
      {groups && Object.keys(groups).length > 0 && (
        <div className="flex flex-wrap gap-2 ">
          {Object.entries(groups).map(([groupName, groupCollection], idx) => (
            <AggregationCollectionDisplay
              key={`${idx}-${groupName}`}
              collectionName={groupName}
              collection={groupCollection}
              depth={depth + 1}
              _collapsed={depth + 1 > 0 ? true : false}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface AggregationCollectionDisplayProps {
  collectionName: string;
  collection: AggregationCollection;
  depth: number;
  _collapsed: boolean;
}

const AggregationCollectionDisplay: React.FC<
  AggregationCollectionDisplayProps
> = ({ collectionName, collection, depth, _collapsed }) => {
  const [collapsed, setCollapsed] = useState(_collapsed);

  return (
    <div
      className={`flex flex-col gap-2 p-3 rounded-lg ${
        depth > 0 ? "bg-foreground" : ""
      } ${collapsed ? "h-10" : ""}`}
    >
      <button
        onClick={() => setCollapsed((prev) => !prev)}
        className="w-full flex justify-start items-start"
      >
        <p className="font-bold text-[11px] text-secondary hover:text-primary transition-all duration-300">
          {collectionName}
        </p>
      </button>
      {!collapsed &&
        Object.entries(collection).map(([fieldName, field], idx) => (
          <AggregationFieldDisplay
            key={`${idx}-${fieldName}`}
            fieldName={fieldName}
            field={field}
            depth={depth}
          />
        ))}
    </div>
  );
};

interface AggregationDisplayProps {
  aggregation: AggregationPayload[];
}

const AggregationDisplay: React.FC<AggregationDisplayProps> = ({
  aggregation,
}) => {
  return (
    <div className="w-full flex flex-col justify-start items-start gap-2">
      {aggregation.map((aggregationPayload, payloadIdx) => (
        <div
          key={`payload-${payloadIdx}`}
          className="w-full flex flex-col gap-2"
        >
          {Object.entries(aggregationPayload).map(
            ([collectionName, collection], idx) => (
              <AggregationCollectionDisplay
                key={`${payloadIdx}-${idx}-${collectionName}`}
                collectionName={collectionName}
                collection={collection}
                depth={0}
                _collapsed={false}
              />
            )
          )}
        </div>
      ))}{" "}
    </div>
  );
};

export default AggregationDisplay;
