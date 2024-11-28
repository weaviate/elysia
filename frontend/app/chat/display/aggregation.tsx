"use client";

import React from "react";
import { AggregationPayload } from "@/app/types";
import KPIDisplay from "./kpi";

interface AggregationDisplayProps {
  aggregation: AggregationPayload[];
}

const AggregationDisplay: React.FC<AggregationDisplayProps> = ({
  aggregation,
}) => {
  return (
    <div className="w-full flex flex-col justify-start items-start gap-2 ">
      {aggregation.map((agg, idx) => (
        <div className="flex flex-col gap-2 w-full">
          {Object.entries(agg).map(([key, value], idx) => (
            <div className="flex flex-col gap-2 w-full">
              <p className="font-bold text-xs text-primary">{key}</p>
              {Object.entries(value).map(([key, value], idx) => (
                <KPIDisplay
                  parent_field={key}
                  values={value.values}
                  type={value.type}
                  key={`${idx}-${key}`}
                />
              ))}
            </div>
          ))}
        </div>
      ))}
      <p className="text-xs text-secondary">{JSON.stringify(aggregation)}</p>
    </div>
  );
};

export default AggregationDisplay;
