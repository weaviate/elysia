import { useCallback } from "react";
import { Handle, Position } from "@xyflow/react";

function DecisionNode({ data }: { data: any }) {
  return (
    <>
      <div draggable className="flex gap-2">
        <div className="bg-background_alt w-[800px] p-8 flex items-center justify-center text-primary rounded-lg">
          <Handle type="target" position={Position.Top} />
          <p className="text-primary font-bold text-2xl">{data.text}</p>
          <Handle type="source" position={Position.Bottom} id="a" />
        </div>
      </div>
    </>
  );
}

export default DecisionNode;
