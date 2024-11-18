"use client";

import React, { useEffect, useMemo } from "react";

import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

import { DecisionPayload } from "../types";
import DecisionNode from "./nodes/decision";
import ReasoningNode from "./nodes/reasoning";
import InstructionNode from "./nodes/instruction";

interface FlowDisplayProps {
  decisions: DecisionPayload[];
}

const FlowDisplay: React.FC<FlowDisplayProps> = ({ decisions }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const createNodes = (decisions: DecisionPayload[]) => {
    const nodes: Node[] = [];
    const centerX = window.innerWidth / 2; // Dynamic center based on window width
    const verticalSpacing = 200; // Consistent vertical spacing between node groups

    decisions.forEach((decision, index) => {
      const baseOffset = index * verticalSpacing * 3; // Multiply by 2 to give more space between decision groups

      // Decision node
      nodes.push({
        id: `${index}-decision`,
        type: "decision",
        draggable: true,
        position: { x: centerX - 150, y: baseOffset }, // Subtract half the typical node width
        data: {
          text: decision.decision,
        },
      });

      // Reasoning node
      nodes.push({
        id: `${index}-reasoning`,
        type: "reasoning",
        draggable: true,
        position: { x: centerX - 150, y: baseOffset + verticalSpacing }, // One spacing unit down
        data: {
          text: decision.reasoning,
        },
      });

      // Instruction node
      nodes.push({
        id: `${index}-instruction`,
        type: "instruction",
        draggable: true,
        position: { x: centerX - 150, y: baseOffset + verticalSpacing * 2 }, // 1.5 spacing units down
        data: {
          text: decision.instruction,
        },
      });
    });
    return nodes;
  };

  const createEdges = (decisions: DecisionPayload[]) => {
    const edges: Edge[] = [];
    decisions.forEach((decision, index) => {
      // Connect decision to reasoning
      edges.push({
        id: `e${index}-decision-reasoning`,
        source: `${index}-decision`,
        target: `${index}-reasoning`,
      });

      // Connect reasoning to instruction
      edges.push({
        id: `e${index}-reasoning-instruction`,
        source: `${index}-reasoning`,
        target: `${index}-instruction`,
      });

      // Connect instruction to next decision if not last
      if (index < decisions.length - 1) {
        edges.push({
          id: `e${index}-instruction-decision`,
          source: `${index}-instruction`,
          target: `${index + 1}-decision`,
        });
      }
    });
    return edges;
  };

  const nodeTypes = useMemo(
    () => ({
      decision: DecisionNode,
      reasoning: ReasoningNode,
      instruction: InstructionNode,
    }),
    []
  );

  useEffect(() => {
    setNodes(createNodes(decisions));
    setEdges(createEdges(decisions));
  }, [decisions]);

  return (
    <div
      className={`w-[80vw] flex justify-start items-start p-4 overflow-scroll transition-all duration-300`}
    >
      <div style={{ width: "100vw", height: "100vh" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          snapGrid={[15, 15]}
          snapToGrid={true}
          fitView
        ></ReactFlow>
      </div>
    </div>
  );
};

export default FlowDisplay;
