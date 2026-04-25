"use client";
import { useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { WorkflowNode, WorkflowEdge } from "@/hooks/use-workflows";
import { Bot, Globe, GitBranch, LoopIcon, Zap, Clock, Timer, Webhook } from "lucide-react";

// Lazy-load ReactFlow to avoid SSR issues
const ReactFlow = dynamic(() => import("@xyflow/react").then((m) => m.ReactFlow), { ssr: false });
const Background = dynamic(() => import("@xyflow/react").then((m) => m.Background), { ssr: false });
const Controls = dynamic(() => import("@xyflow/react").then((m) => m.Controls), { ssr: false });
const MiniMap = dynamic(() => import("@xyflow/react").then((m) => m.MiniMap), { ssr: false });

const NODE_TYPES_CONFIG = [
  { type: "trigger", icon: Zap, label: "Trigger", color: "violet" },
  { type: "agent", icon: Bot, label: "AI Agent", color: "blue" },
  { type: "http", icon: Globe, label: "HTTP Request", color: "green" },
  { type: "condition", icon: GitBranch, label: "Condition", color: "yellow" },
  { type: "loop", icon: Timer, label: "Loop", color: "orange" },
  { type: "transform", icon: Zap, label: "Transform", color: "pink" },
  { type: "delay", icon: Clock, label: "Delay", color: "gray" },
];

interface WorkflowBuilderProps {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  onChange: (nodes: WorkflowNode[], edges: WorkflowEdge[]) => void;
}

export function WorkflowBuilder({ nodes, edges, onChange }: WorkflowBuilderProps) {
  const [rfNodes, setRfNodes] = useState(
    nodes.map((n) => ({ ...n, data: { ...n.data, label: n.data.label || n.type } }))
  );
  const [rfEdges, setRfEdges] = useState(edges);
  const idCounter = useRef(nodes.length + 1);

  const addNode = (type: string) => {
    const id = `node_${idCounter.current++}`;
    const newNode = {
      id,
      type,
      position: { x: 100 + Math.random() * 400, y: 100 + Math.random() * 200 },
      data: { label: type.charAt(0).toUpperCase() + type.slice(1) },
    };
    const updated = [...rfNodes, newNode];
    setRfNodes(updated);
    onChange(updated, rfEdges);
  };

  const onConnect = useCallback(
    (connection: { source: string; target: string }) => {
      const edge: WorkflowEdge = {
        id: `e_${connection.source}_${connection.target}`,
        source: connection.source,
        target: connection.target,
      };
      const updated = [...rfEdges, edge];
      setRfEdges(updated);
      onChange(rfNodes, updated);
    },
    [rfNodes, rfEdges, onChange]
  );

  const onNodesChange = useCallback(
    (changes: unknown[]) => {
      // Apply position changes
      import("@xyflow/react").then(({ applyNodeChanges }) => {
        const updated = applyNodeChanges(changes as Parameters<typeof applyNodeChanges>[0], rfNodes);
        setRfNodes(updated);
        onChange(updated, rfEdges);
      });
    },
    [rfNodes, rfEdges, onChange]
  );

  return (
    <div className="flex h-full">
      {/* Node palette */}
      <div className="w-44 border-r border-white/5 bg-[#0f0f17] flex flex-col gap-1 p-2 overflow-y-auto">
        <p className="text-xs text-muted-foreground px-1 py-1">Node Types</p>
        {NODE_TYPES_CONFIG.map(({ type, icon: Icon, label, color }) => (
          <button
            key={type}
            onClick={() => addNode(type)}
            className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/5 text-xs text-muted-foreground hover:text-white text-left"
          >
            <div className={`w-5 h-5 rounded flex items-center justify-center bg-${color}-500/20`}>
              <Icon className={`w-3 h-3 text-${color}-400`} />
            </div>
            {label}
          </button>
        ))}
      </div>

      {/* Canvas */}
      <div className="flex-1 h-full bg-[#0d0d0f]">
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          onConnect={onConnect as unknown as Parameters<typeof ReactFlow>[0]["onConnect"]}
          onNodesChange={onNodesChange as unknown as Parameters<typeof ReactFlow>[0]["onNodesChange"]}
          fitView
          style={{ background: "#0d0d0f" }}
          defaultEdgeOptions={{ animated: true, style: { stroke: "#7c3aed" } }}
        >
          <Background color="#333" gap={20} />
          <Controls className="!bg-[#13131a] !border-white/10" />
          <MiniMap className="!bg-[#0f0f17]" nodeColor="#7c3aed" maskColor="rgba(0,0,0,0.7)" />
        </ReactFlow>
      </div>
    </div>
  );
}
