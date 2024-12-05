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
    | "warning"
    | "tree_update";
  conversation_id: string;
  id: string;
  query_id: string;
  payload:
    | ResultPayload
    | TextPayload
    | ErrorPayload
    | ResponsePayload
    | TreeUpdatePayload;
};

export type ResponsePayload = {
  type: "response" | "summary" | "code";
  /* eslint-disable @typescript-eslint/no-explicit-any */
  metadata: any;
  objects: TextPayload[] | SummaryPayload[] | CodePayload[];
};

export type ResultPayload = {
  type:
    | "text"
    | "ticket"
    | "message"
    | "conversation"
    | "ecommerce"
    | "epic_generic"
    | "boring_generic"
    | "aggregation"
    | "mapped"
    | "document";
  /* eslint-disable @typescript-eslint/no-explicit-any */
  metadata: any;
  code: CodePayload;
  objects:
    | string[]
    | Ticket[]
    | ConversationMessage[]
    | ConversationMessage[][] // A list of lists of ConversationMessages
    | Ecommerce[]
    | { [key: string]: string }[]
    | AggregationPayload[]
    | EpicGeneric[]
    | DocumentPayload[];
};

export type AggregationPayload = {
  [key: string]: AggregationCollection;
};

export type AggregationCollection = {
  [key: string]: AggregationField;
};

export type AggregationField = {
  type: "text" | "number";
  values: AggregationValue[];
  groups: { [key: string]: AggregationCollection };
};

export type AggregationValue = {
  value: string | number;
  field: string | null;
  aggregation: "count" | "sum" | "avg" | "minimum" | "maximum" | "mean";
};

export type DocumentPayload = {
  uuid: string;
  summary?: string;
  title: string;
  author: string;
  date: string;
  content: string;
  category: string;
  chunk_spans: [number, number][];
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

export type ObjectRelevancyPayload = {
  conversation_id: string;
  any_relevant: boolean;
  error: string;
};

export type EpicGeneric = {
  uuid: string;
  summary?: string;
  title: string;
  subtitle: string;
  content: string;
  url: string;
  id: string;
  author: string;
  timestamp: string;
  category: string;
  subcategory: string;
  tags: string[];
};

export type Ecommerce = {
  subcategory: string;
  description: string;
  reviews: string[] | number;
  collection: string;
  tags: string[];
  sizes: string[];
  product_id: string;
  image: string;
  url: string;
  rating: number;
  price: number;
  category: string;
  colors: string[];
  brand: string;
  name: string;
  id: string;
  uuid: string;
  summary?: string;
};

export type Ticket = {
  uuid: string;
  summary?: string;
  updated_at: string;
  title: string;
  subtitle: string;
  content: string;
  created_at: string;
  author: string;
  url: string;
  status: string;
  id: string;
  tags: string[];
  comments: number | string[];
};

export type ConversationMessage = {
  uuid: string;
  summary?: string;
  relevant: boolean;
  conversation_id: number;
  message_id: string;
  author: string;
  content: string;
  timestamp: string;
};

export type TreeUpdatePayload = {
  node: string;
  decision: string;
  tree_index: number;
  reasoning: string;
  reset: boolean;
};

export type Conversation = {
  enabled_collections: { [key: string]: boolean };
  id: string;
  name: string;
  tree: DecisionTreeNode[];
  base_tree: DecisionTreeNode | null;
  queries: { [key: string]: Query };
  current: string;
};

export type Query = {
  id: string;
  query: string;
  messages: Message[];
  index: number;
};

export type DecisionTreePayload = {
  conversation_id: string;
  error: string;
  tree: DecisionTreeNode;
};

export type DecisionTreeNode = {
  name: string;
  id: string;
  description: string;
  instruction: string;
  reasoning: string;
  options: { [key: string]: DecisionTreeNode };
  choosen?: boolean;
  blocked?: boolean;
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
  id: uuidv4(),
  name: "New Conversation",
  enabled_collections: {},
  tree: [],
  base_tree: null,
  current: "",
  queries: {},
};
