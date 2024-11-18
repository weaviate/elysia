"use client";

import React, { useCallback, useEffect, useMemo } from "react";

import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  ConnectionLineType,
} from "@xyflow/react";
import dagre from "dagre";

import "@xyflow/react/dist/style.css";

import DecisionNode from "./nodes/decision";
import { DecisionTreeNode } from "../types";

interface FlowDisplayProps {
  currentTree: DecisionTreeNode | null;
}

const FlowDisplay: React.FC<FlowDisplayProps> = ({ currentTree }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const nodeTypes = useMemo(
    () => ({
      decision: DecisionNode,
    }),
    []
  );

  // Dagre graph setup for layout
  const dagreGraph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));

  const nodeWidth = 300;
  const nodeHeight = 100;

  const getLayoutedElements = (
    nodes: Node[],
    edges: Edge[],
    direction = "TB"
  ) => {
    const isHorizontal = direction === "LR";
    dagreGraph.setGraph({
      rankdir: direction,
      ranksep: 100, // Vertical spacing between nodes (default is 50)
      nodesep: 100, // Horizontal spacing between nodes (default is 50)
    });

    nodes.forEach((node) => {
      dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
    });

    edges.forEach((edge) => {
      dagreGraph.setEdge(edge.source, edge.target);
    });

    dagre.layout(dagreGraph);

    const newNodes = nodes.map((node) => {
      const nodeWithPosition = dagreGraph.node(node.id);
      const newNode = {
        ...node,
        targetPosition: isHorizontal ? "left" : "top",
        sourcePosition: isHorizontal ? "right" : "bottom",
        // We are shifting the dagre node position (anchor=center center) to the top left
        // so it matches the React Flow node anchor point (top left).
        position: {
          x: nodeWithPosition.x - nodeWidth / 2,
          y: nodeWithPosition.y - nodeHeight / 2,
        },
      };

      return newNode;
    });

    return { nodes: newNodes, edges };
  };

  const createNodesEdges = (tree: DecisionTreeNode) => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    let idCounter = 0;
    const getId = () => `node-${idCounter++}`;

    const traverse = (
      node: DecisionTreeNode,
      parentId: string | null = null
    ) => {
      const nodeId = getId();

      // Create decision node
      nodes.push({
        id: nodeId,
        type: "decision",
        data: {
          text: node.name,
          description: node.description,
          choosen: node.choosen,
          instruction: node.instruction,
        },
        position: { x: 0, y: 0 },
      });

      if (parentId) {
        edges.push({
          id: `edge-${parentId}-${nodeId}`,
          source: parentId,
          target: nodeId,
          type: "smoothstep",
          animated: node.choosen,
        });
      }

      if (node.options && Object.keys(node.options).length > 0) {
        Object.keys(node.options).forEach((option) => {
          traverse(node.options[option], nodeId);
        });
      }
    };

    traverse(tree);

    return getLayoutedElements(nodes, edges);
  };

  useEffect(() => {
    if (currentTree) {
      const { nodes, edges } = createNodesEdges(currentTree);
      setNodes(nodes as Node[]);
      setEdges(edges);
    } else {
      setNodes([]);
      setEdges([]);
    }
  }, [currentTree]);

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
          connectionLineType={ConnectionLineType.SmoothStep}
          onEdgesChange={onEdgesChange}
          fitView
          nodesDraggable={false}
          draggable={false}
        ></ReactFlow>
      </div>
    </div>
  );
};

export default FlowDisplay;
