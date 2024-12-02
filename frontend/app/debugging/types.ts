export type DebugResponse = {
  [key: string]: any;
};

export type DebugModel = {
  model: string;
  chat: DebugMessage[][];
};

export type DebugMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};
