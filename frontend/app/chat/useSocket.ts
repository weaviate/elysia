import { useEffect, useRef, useState } from "react";
import {
  Message,
  TextMessage,
  ErrorMessage,
  TicketMessage,
  DecisionMessage,
  DecisionPayload,
} from "../types";
import { getWebsocketHost } from "../api";
import { v4 as uuidv4 } from "uuid";

export function useSocket(
  addMessageToConversation: (
    message: Message[],
    conversation_id: string
  ) => void,
  setConversationStatus: (status: string, conversationId: string) => void,
  setAllConversationStatuses: (status: string) => void
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
        if (message.type === "decision") {
          const payload = message.payload as DecisionPayload;
          setConversationStatus(payload.decision, message.conversation_id);
        }
        const newMessage = { ...message, collapsed: true, id: uuidv4() };
        addMessageToConversation([newMessage], newMessage.conversation_id);
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
    conversation_id: string
  ) => {
    //fakeQuery(conversation_id);
    setConversationStatus("Thinking...", conversation_id);
    socket?.send(JSON.stringify({ user_id, query, conversation_id }));
  };

  const fakeQuery = (conversation_id: string) => {
    // This is fake data for now
    const text_message: Message = { ...TextMessage, conversation_id };
    const error_message: Message = { ...ErrorMessage, conversation_id };
    const ticket_message: Message = { ...TicketMessage, conversation_id };
    const decision_message: Message = {
      ...DecisionMessage,
      conversation_id,
    };

    setConversationStatus("Thinking...", conversation_id);

    setTimeout(() => {
      setConversationStatus("Querying...", conversation_id);
    }, 2000);

    setTimeout(() => {
      setConversationStatus("Collecting...", conversation_id);
    }, 4000);

    setTimeout(() => {
      addMessageToConversation(
        [text_message, error_message, ticket_message, decision_message],
        conversation_id
      );
      setConversationStatus("", conversation_id);
    }, 8000);
  };

  return { socketOnline, sendQuery };
}
