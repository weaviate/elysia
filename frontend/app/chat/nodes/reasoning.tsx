import { Handle, Position } from "@xyflow/react";

/* eslint-disable @typescript-eslint/no-explicit-any */
function ReasoningNode({ data }: { data: any }) {
  return (
    <>
      <div className="bg-foreground w-[800px] p-4 flex items-center justify-center rounded-lg">
        <Handle type="target" position={Position.Top} />
        <p className="text-primary text-xs">{data.text}</p>
        <Handle type="source" position={Position.Bottom} id="a" />
      </div>
    </>
  );
}

export default ReasoningNode;
