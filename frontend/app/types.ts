import { v4 as uuidv4 } from "uuid";

export type Message = {
  type: "Text" | "Error" | "User" | "Ticket" | "TreeUpdate";
  conversation_id: string;
  payload: string | TicketPayload | TreeUpdatePayload;
};

export type TicketPayload = {
  issue_id: string;
  issue_updated_at: string;
  issue_title: string;
  issue_content: string;
  issue_created_at: string;
};

export type TreeUpdatePayload = {
  id: string;
  type: string;
  instruction: string;
  tree: TreeNode[];
};

export type TreeNode = { id: string; instruction: string };

export type Conversation = {
  messages: Message[];
  id: string;
  name: string;
  tree: TreeNode[];
  current: string;
};

export const initialConversation: Conversation = {
  messages: [],
  id: uuidv4(),
  name: "New Conversation",
  tree: [],
  current: "",
};

export const ErrorMessage: Message = {
  type: "Error",
  conversation_id: "",
  payload: "This is an **error message**",
};

export const TextMessage: Message = {
  type: "Text",
  conversation_id: "",
  payload: "This is a **text message**",
};

export const TicketMessage: Message = {
  type: "Ticket",
  conversation_id: "",
  payload: {
    issue_id: "2377991602",
    issue_updated_at: "2024-07-02T10:48:53Z",
    issue_title: "Addition of ability to read .docx files",
    issue_content:
      "This pull request is a new feature to add the ability to read .docx files using the BasicReader. It does pull in a new dependency as well (python-docx).",
    issue_created_at: "2024-06-27T12:12:19Z",
  },
};

export const TreeUpdateMessage: Message = {
  type: "TreeUpdate",
  conversation_id: "",
  payload: {
    id: "Node-1",
    type: "Querying",
    instruction: "Querying GitHub Collection",
    tree: [{ id: "Initial", instruction: "Starting Tree" }],
  },
};
