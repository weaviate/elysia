"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Ticket } from "../../types";
import MarkdownMessageDisplay from "./markdown";

interface TicketMessageDisplayProps {
  ticket: Ticket;
}

const TicketMessageDisplay: React.FC<TicketMessageDisplayProps> = ({
  ticket,
}) => {
  const [ticketCollapsed, setTicketCollapsed] = useState(true);

  return (
    <div
      className="flex flex-col w-full cursor-pointer hover:bg-foreground_alt bg-foreground p-5 rounded-xl flex-grow justify-start items-start gap-2 chat-animation"
      onClick={() => setTicketCollapsed((prev) => !prev)}
    >
      <div className="flex flex-col gap-2">
        <div className="flex flex-col items-start gap-1">
          <p className="text-xs font-light text-secondary">
            {ticket.issue_created_at}
          </p>
          <p className="text-primary text-base font-bold">
            {ticket.issue_title}
          </p>
        </div>
        <div
          className={`text-primary overflow-scroll text-sm gap-5 max-w-[70vw] flex flex-col text-wrap ${
            ticketCollapsed ? "max-h-[8vh]" : "max-h-[32vh]"
          }`}
        >
          <MarkdownMessageDisplay text={ticket.issue_content} />
        </div>
      </div>
    </div>
  );
};

export default TicketMessageDisplay;
