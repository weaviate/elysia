import { getDebug } from "./api";
import { DebugResponse } from "./types";

export function useDebug(user_id: string) {
  const fetchDebug = async (
    conversation_id: string
  ): Promise<DebugResponse> => {
    const debug = await getDebug(user_id, conversation_id);
    return debug;
  };

  return {
    fetchDebug,
  };
}
