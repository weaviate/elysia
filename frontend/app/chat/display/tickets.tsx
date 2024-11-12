"use client";

import React, { useState } from "react";

import { Message, ResultPayload, Ticket } from "../../types";
import { TbDots } from "react-icons/tb";
import TicketMessageDisplay from "./ticket";

interface TicketsDisplayProps {
  message: Message;
  toggleMessageCollapsed: (conversation_id: string, message_id: string) => void;
}

const TicketsDisplay: React.FC<TicketsDisplayProps> = ({
  message,
  toggleMessageCollapsed,
}) => {
  const [ticketCollapsed, setTicketCollapsed] = useState(true);
  const payload = message.payload as ResultPayload;
  const tickets = payload.objects as Ticket[];

  if (tickets.length === 0) return null;

  return (
    <div className="w-full flex flex-col justify-start items-start gap-3">
      {payload.metadata["collection_name"] && (
        <div className="w-full flex flex-col justify-start items-start gap-2 mb-2">
          <p className="text-xs transition-all duration-300 cursor-pointer hover:text-primary hover:border-white text-secondary border border-secondary rounded-lg p-2">
            {payload.metadata["collection_name"]}
          </p>
        </div>
      )}
      <div className="flex flex-col w-full justify-start items-start gap-2">
        {(message.collapsed ? tickets.slice(0, 3) : tickets).map(
          (ticket, idx) => (
            <TicketMessageDisplay
              key={`${idx}-${message.id}`}
              ticket={ticket}
            />
          )
        )}
      </div>
      {tickets.length > 3 && (
        <div className="flex w-full justify-center items-center">
          <button
            className="btn w-1/5 bg-background text-primary text-xs items-center justify-center"
            onClick={() => {
              toggleMessageCollapsed(message.conversation_id, message.id || "");
            }}
          >
            <TbDots className="text-primary text-xs" />
            {message.collapsed ? "Show All " + tickets.length : "Show Less"}
          </button>
        </div>
      )}
    </div>
  );
};

export default TicketsDisplay;
