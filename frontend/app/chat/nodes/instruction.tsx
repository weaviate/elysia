import { useCallback } from "react";
import { Handle, Position } from "@xyflow/react";

function InstructionNode({ data }: { data: any }) {
  return (
    <>
      <div className="bg-foreground_alt  w-[800px] p-4 flex items-center justify-center rounded-lg">
        <Handle type="target" position={Position.Top} />
        <p className="text-primary font-light text-xs">{data.text}</p>
        <Handle type="source" position={Position.Bottom} id="a" />
      </div>
    </>
  );
}

export default InstructionNode;
