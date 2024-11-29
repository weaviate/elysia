"use client";

import React from "react";
import {
  AggregationCollection,
  AggregationField,
  AggregationPayload,
} from "@/app/types";
import KPIDisplay from "./kpi";

interface AggregationFieldDisplayProps {
  fieldName: string;
  field: AggregationField;
}

const AggregationFieldDisplay: React.FC<AggregationFieldDisplayProps> = ({
  fieldName,
  field,
}) => {
  const { type, values, groups } = field;
  return (
    <div className="flex flex-col gap-6 w-full  ">
      <KPIDisplay parent_field={fieldName} values={values} type={type} />
      {groups && Object.keys(groups).length > 0 && (
        <div className="flex flex-wrap gap-2 bg-background_alt p-3 rounded-lg">
          {Object.entries(groups).map(([groupName, groupCollection], idx) => (
            <AggregationCollectionDisplay
              key={`${idx}-${groupName}`}
              collectionName={groupName}
              collection={groupCollection}
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
}

const AggregationCollectionDisplay: React.FC<
  AggregationCollectionDisplayProps
> = ({ collectionName, collection }) => {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-bold text-[11px] text-secondary">{collectionName}</p>
      {Object.entries(collection).map(([fieldName, field], idx) => (
        <AggregationFieldDisplay
          key={`${idx}-${fieldName}`}
          fieldName={fieldName}
          field={field}
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
              />
            )
          )}
        </div>
      ))}{" "}
    </div>
  );
};

export default AggregationDisplay;
