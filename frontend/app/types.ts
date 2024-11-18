import { v4 as uuidv4 } from "uuid";

export type Message = {
  type:
    | "result"
    | "error"
    | "text"
    | "User"
    | "decision"
    | "status"
    | "completed"
    | "warning";
  conversation_id: string;
  id: string;
  collapsed?: boolean; //added for ticket display
  payload:
    | ResultPayload
    | DecisionPayload
    | TextPayload
    | ErrorPayload
    | ResponsePayload;
};

export type ResponsePayload = {
  type: "response" | "summary" | "code";
  /* eslint-disable @typescript-eslint/no-explicit-any */
  metadata: any;
  objects: TextPayload[] | SummaryPayload[] | CodePayload[];
};

export type ResultPayload = {
  type: "text" | "ticket" | "message" | "conversation";
  /* eslint-disable @typescript-eslint/no-explicit-any */
  metadata: any;
  objects:
    | string[]
    | Ticket[]
    | ConversationMessage[]
    | ConversationMessage[][]; // A list of lists of ConversationMessages
};

export type TextPayload = {
  text: string;
};

export type SummaryPayload = {
  text: string;
  title: string;
};

export type CodePayload = {
  language: string;
  title: string;
  text: string;
};

export type ErrorPayload = {
  error: string;
};

export type Ticket = {
  uuid: string;
  summary?: string;
  issue_id: string;
  issue_updated_at: string;
  issue_title: string;
  issue_content: string;
  issue_created_at: string;
  issue_author: string;
  issue_url: string;
  issue_labels: string[];
  issue_state: string;
  issue_comments: number;
};

export type ConversationMessage = {
  uuid: string;
  relevant: boolean;
  conversation_id: number;
  message_index: number;
  message_author: string;
  message_content: string;
  message_timestamp: string;
};

export type DecisionPayload = {
  id: string;
  decision: string;
  reasoning: string;
  instruction: string;
  /* eslint-disable @typescript-eslint/no-explicit-any */
  tree: any[];
};

export type Conversation = {
  messages: Message[];
  decisions: DecisionPayload[];
  enabled_collections: { [key: string]: boolean };
  id: string;
  name: string;
  /* eslint-disable @typescript-eslint/no-explicit-any */
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
  /* eslint-disable @typescript-eslint/no-explicit-any */
  items: { [key: string]: any }[];
  error: string;
};

export type NERResponse = {
  text: string;
  entity_spans: [number, number][];
  noun_spans: [number, number][];
};

export type TitleResponse = {
  title: string;
  error: string;
};

export type ErrorResponse = {
  error: string;
};

// Example Objects

export const initialConversation: Conversation = {
  messages: [],
  id: uuidv4(),
  decisions: [],
  name: "New Conversation",
  enabled_collections: {},
  tree: [],
  current: "",
};
