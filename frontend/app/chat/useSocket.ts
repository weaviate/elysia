import { useEffect, useState } from "react";
import { Message, TextPayload } from "../types";
import { getWebsocketHost } from "../api";

export function useSocket(
  addMessageToConversation: (
    message: Message[],
    conversation_id: string,
    query_id: string
  ) => void,
  setConversationStatus: (status: string, conversationId: string) => void,
  setAllConversationStatuses: (status: string) => void,
  updateTree: (message: Message) => void
) {
  const [socketOnline, setSocketOnline] = useState(false);
  const [socket, setSocket] = useState<WebSocket>();

  const [reconnect, setReconnect] = useState(false);

  useEffect(() => {
    setReconnect(true);
  }, []);

  useEffect(() => {
    const socketHost = getWebsocketHost();
    const localSocket = new WebSocket(socketHost);

    localSocket.onopen = () => {
      setSocketOnline(true);
      console.log("Socket opened");
    };

    localSocket.onmessage = (event) => {
      try {
        const message: Message = JSON.parse(event.data);
        console.log(message.type);
        if (message.type === "status") {
          const payload = message.payload as TextPayload;
          setConversationStatus(payload.text, message.conversation_id);
        } else if (message.type === "completed") {
          setConversationStatus("", message.conversation_id);
        } else if (message.type === "tree_update") {
          updateTree(message);
        } else {
          console.log(message);
          addMessageToConversation(
            [message],
            message.conversation_id,
            message.query_id
          );
        }
      } catch (error) {
        console.error(error);
      }
    };

    localSocket.onerror = (error) => {
      console.log(error);
      setSocketOnline(false);
      setAllConversationStatuses("");
      setReconnect((prev) => !prev);
    };

    localSocket.onclose = () => {
      setSocketOnline(false);
      setAllConversationStatuses("");
      console.log("Socket closed");
      setReconnect((prev) => !prev);
    };

    setSocket(localSocket);

    return () => {
      if (localSocket.readyState !== WebSocket.CLOSED) {
        localSocket.close();
      }
    };
  }, [reconnect]);

  const sendQuery = (
    user_id: string,
    query: string,
    conversation_id: string,
    query_id: string,
    route?: string
  ) => {
    const mimick = route ? true : false;
    setConversationStatus("Thinking...", conversation_id);
    socket?.send(
      JSON.stringify({
        user_id,
        query,
        conversation_id,
        query_id,
        route,
        mimick,
      })
    );
  };

  return { socketOnline, sendQuery };
}
