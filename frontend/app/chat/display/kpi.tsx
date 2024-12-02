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
    <div className="flex flex-wrap gap-2">
      {values.map((value, idx) => (
        <div
          key={`${idx}-${value.value}`}
          className={`cursor-pointer hover:bg-foreground_alt flex flex-col gap-2 min-w-[100px] aspect-square items-center justify-center border-secondary ${
            idx % 2 === 0 ? "bg-background" : "bg-background_alt"
          } border rounded-lg p-2 shadow-lg`}
        >
          <p className="font-bold text-[11px] text-secondary w-full">
            {value.field ? value.field : parent_field}
          </p>
          <p className="text-xl font-bold [text-shadow:0_0_10px_rgba(0,0,0,0.5)]">
            {type === "number" ? Number(value.value).toFixed(2) : value.value}
          </p>
          <p className="text-[11px] text-secondary">
            {value.aggregation.toUpperCase()}
          </p>
        </div>
      ))}
    </div>
  );
};

export default KPIDisplay;
