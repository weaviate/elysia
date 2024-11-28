"use client";

import React, { useEffect, useState } from "react";

import { Message, ResultPayload, Ticket } from "../../types";
import TicketMessageDisplay from "./ticket";

interface TicketsDisplayProps {
  message: Message;
}

const TicketsDisplay: React.FC<TicketsDisplayProps> = ({ message }) => {
  const payload = message.payload as ResultPayload;
  const tickets = payload.objects as Ticket[];

  const [ticketCollapsed, setTicketCollapsed] = useState(false);

  useEffect(() => {
    if (tickets.length > 3) {
      setTicketCollapsed(true);
    }
  }, [tickets]);

  if (tickets.length === 0) return null;

  return (
    <div className="w-full flex flex-col justify-start items-start gap-3 ">
      <div className="flex flex-col w-full justify-start items-start gap-3">
        {(ticketCollapsed ? tickets.slice(0, 3) : tickets).map(
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
              setTicketCollapsed((prev) => !prev);
            }}
          >
            {ticketCollapsed
              ? "Show All " + tickets.length + " Tickets"
              : "Show Less"}
          </button>
        </div>
      )}
    </div>
  );
};

export default TicketsDisplay;
