import { useEffect, useRef, useState } from "react";
import {
  Message,
  TextMessage,
  ErrorMessage,
  TicketMessage,
  TreeUpdateMessage,
  TreeUpdatePayload,
} from "../types";
import { getWebsocketHost } from "../api";

export function useSocket(
  addMessageToConversation: (
    message: Message[],
    conversation_id: string
  ) => void,
  setConversationStatus: (status: string, conversationId: string) => void
) {
  const [socketOnline, setSocketOnline] = useState(false);
  const [socket, setSocket] = useState<WebSocket>();

  const [reconnect, setReconnect] = useState(false);

  useEffect(() => {
    //Uncomment once Backend is ready
    //setReconnect(true);
  }, []);

  useEffect(() => {
    const socketHost = getWebsocketHost();
    const localSocket = new WebSocket(socketHost);

    localSocket.onopen = () => {
      setSocketOnline(true);
    };

    localSocket.onmessage = (event) => {
      try {
        const message: Message = JSON.parse(event.data);
        if (message.type === "TreeUpdate") {
          const payload = message.payload as TreeUpdatePayload;
          setConversationStatus(payload.type, message.conversation_id);
        }
        addMessageToConversation([message], message.conversation_id);
      } catch (error) {
        console.error(error);
      }
    };

    localSocket.onerror = (error) => {
      console.error(error);
      setSocketOnline(false);
      setReconnect((prev) => !prev);
    };

    localSocket.onclose = () => {
      setSocketOnline(false);
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
    conversation_id: string
  ) => {
    // This is fake data for now
    const text_message: Message = { ...TextMessage, conversation_id };
    const error_message: Message = { ...ErrorMessage, conversation_id };
    const ticket_message: Message = { ...TicketMessage, conversation_id };
    const tree_update_message: Message = {
      ...TreeUpdateMessage,
      conversation_id,
    };

    setConversationStatus("Thinking...", conversation_id);

    setTimeout(() => {
      setConversationStatus("Querying...", conversation_id);
    }, 1000);

    setTimeout(() => {
      setConversationStatus("Collecting...", conversation_id);
    }, 2000);

    setTimeout(() => {
      addMessageToConversation(
        [text_message, error_message, ticket_message, tree_update_message],
        conversation_id
      );
    }, 4000);

    //socket?.send(JSON.stringify({ user_id, query, conversation_id }));
  };

  return { socketOnline, sendQuery };
}
