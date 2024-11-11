import { v4 as uuidv4 } from "uuid";

export type Message = {
  type: "result" | "error" | "User" | "decision" | "status" | "completed";
  conversation_id: string;
  id?: string;
  collapsed?: boolean; //added for ticket display
  payload: ResultPayload | DecisionPayload | TextPayload;
};

export type ResultPayload = {
  type: "text" | "ticket";
  metadata: any;
  objects: string[] | Ticket[];
};

export type TextPayload = {
  text: string;
};

export type Ticket = {
  issue_id: string;
  issue_updated_at: string;
  issue_title: string;
  issue_content: string;
  issue_created_at: string;
};

export type DecisionPayload = {
  id: string;
  decision: string;
  reasoning: string;
  instruction: string;
  tree: any[];
};

export type Conversation = {
  messages: Message[];
  id: string;
  name: string;
  tree: any[];
  current: string;
};

export type CollectionPayload = {
  collections: Collection[];
  error: string;
};

export type Collection = {
  name: string;
  total: number;
  vectorizer: string;
};

export type CollectionData = {
  properties: { [key: string]: string };
  items: { [key: string]: any }[];
  error: string;
};

// Example Objects

export const initialConversation: Conversation = {
  messages: [],
  id: uuidv4(),
  name: "New Conversation",
  tree: [],
  current: "",
};

export const ErrorMessage: Message = {
  type: "error",
  conversation_id: "",
  payload: {
    type: "text",
    metadata: {},
    objects: ["This is an **error message**"],
  },
};

export const TextMessage: Message = {
  type: "result",
  conversation_id: "",
  payload: {
    type: "text",
    metadata: {},
    objects: ["This is a **text message**"],
  },
};

export const TicketMessage: Message = {
  type: "result",
  conversation_id: "",
  payload: {
    type: "ticket",
    metadata: {},
    objects: [
      {
        issue_id: "2377991602",
        issue_updated_at: "2024-07-02T10:48:53Z",
        issue_title: "Addition of ability to read .docx files",
        issue_content:
          "This pull request is a new feature to add the ability to read .docx files using the BasicReader. It does pull in a new dependency as well (python-docx).",
        issue_created_at: "2024-06-27T12:12:19Z",
      },
    ],
  },
};

export const DecisionMessage: Message = {
  type: "decision",
  conversation_id: "",
  payload: {
    id: "Node-1",
    decision: "Querying GitHub Collection",
    reasoning: "User requested to query the GitHub Collection",
    instruction: "Querying GitHub Collection",
    tree: [{ id: "Initial", instruction: "Starting Tree" }],
  },
};
