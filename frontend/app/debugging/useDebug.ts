import { useEffect, useState } from "react";
import { getDebug } from "./api";
import { DebugResponse } from "./types";

export function useDebug(user_id: string, conversation_id: string) {
  const [debug, setDebug] = useState<DebugResponse | null>(null);

  useEffect(() => {
    fetchDebug();
  }, []);

  const fetchDebug = async () => {
    const debug = await getDebug(user_id, conversation_id);
    setDebug(debug);
  };

  return {
    debug,
    fetchDebug,
  };
}
