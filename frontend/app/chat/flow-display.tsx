"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Node,
  Edge,
  BackgroundVariant,
  useReactFlow,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

import { DecisionPayload } from "../types";
import DecisionNode from "./nodes/decision";
import ReasoningNode from "./nodes/reasoning";
import InstructionNode from "./nodes/instruction";

import Dagre from "@dagrejs/dagre";

const getLayoutedElements = (
  nodes: Node[],
  edges: Edge[],
  options: { direction: string }
) => {
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: options.direction });

  edges.forEach((edge) => g.setEdge(edge.source, edge.target));
  nodes.forEach((node) =>
    g.setNode(node.id, {
      ...node,
      width: node.measured?.width ?? 0,
      height: node.measured?.height ?? 0,
    })
  );

  Dagre.layout(g);

  return {
    nodes: nodes.map((node) => {
      const position = g.node(node.id);
      // We are shifting the dagre node position (anchor=center center) to the top left
      // so it matches the React Flow node anchor point (top left).
      const x = position.x - (node.measured?.width ?? 0) / 2;
      const y = position.y - (node.measured?.height ?? 0) / 2;

      return { ...node, position: { x, y } };
    }),
    edges,
  };
};

interface FlowDisplayProps {
  decisions: DecisionPayload[];
}

const FlowDisplay: React.FC<FlowDisplayProps> = ({ decisions }) => {
  const { fitView } = useReactFlow();
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

  const onLayout = useCallback(
    (direction: string) => {
      const layouted = getLayoutedElements(nodes, edges, { direction });

      setNodes([...layouted.nodes]);
      setEdges([...layouted.edges]);

      window.requestAnimationFrame(() => {
        fitView();
      });
    },
    [nodes, edges]
  );

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
