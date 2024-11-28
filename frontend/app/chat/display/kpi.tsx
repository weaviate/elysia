"use client";

import React from "react";
import { AggregationValue } from "@/app/types";
interface KPIDisplayProps {
  parent_field: string;
  values: AggregationValue[];
  type: "number" | "text";
}

const KPIDisplay: React.FC<KPIDisplayProps> = ({
  parent_field,
  values,
  type,
}) => {
  return (
    <div className="w-full flex flex-grow gap-2 ">
      {values.map((value, idx) => (
        <div
          key={`${idx}-${value.value}`}
          className="flex flex-col gap-2 w-full items-center justify-center border-secondary border rounded-lg p-2 shadow-lg"
        >
          <p className="font-bold text-xs text-secondary w-full">
            {value.field ? value.field : parent_field}
          </p>
          <p
            className="text-2xl font-bold shadow-lg"
            key={`${idx}-${value.value}`}
          >
            {type === "number" ? Number(value.value).toFixed(2) : value.value}
          </p>
          <p className="text-xs text-secondary">
            {value.aggregation.toUpperCase()}
          </p>
        </div>
      ))}
    </div>
  );
};

export default KPIDisplay;
