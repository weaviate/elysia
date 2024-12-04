"use client";

import { EpicGeneric } from "@/app/types";
import EpicGenericContent from "./epic_generic_content";

interface EpicGenericDisplayProps {
  payload: EpicGeneric[];
}

const EpicGenericDisplay: React.FC<EpicGenericDisplayProps> = ({ payload }) => {
  return (
    <div className="w-full flex overflow-x-scroll justify-start max-h-[50vh] p-4 items-start gap-3">
      {payload.map((p, index) => (
        <div
          key={`${p.uuid}-${index}`}
          className="p-8 bg-foreground shadow-xl rounded-lg w-[50vw] flex-shrink-0 flex flex-col gap-2"
        >
          <p className="text-sm font-medium">{p.title}</p>
          <p className="text-xs text-secondary">{p.subtitle}</p>
          <EpicGenericContent _text={p.content} />
        </div>
      ))}
    </div>
  );
};

export default EpicGenericDisplay;
