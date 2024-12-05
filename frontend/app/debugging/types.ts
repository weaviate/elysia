export type DebugResponse = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
