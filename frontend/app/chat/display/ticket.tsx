"use client";

import React, { useState } from "react";
import { Ticket } from "../../types";
import MarkdownMessageDisplay from "./markdown";
import { FaExternalLinkAlt } from "react-icons/fa";

interface TicketMessageDisplayProps {
  ticket: Ticket;
}

const TicketMessageDisplay: React.FC<TicketMessageDisplayProps> = ({
  ticket,
}) => {
  const [ticketCollapsed, setTicketCollapsed] = useState(true);

  const formatDate = (date: string) => {
    const dateObj = new Date(date);
    return dateObj.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const openLink = () => {
    window.open(ticket.issue_url, "_blank");
  };

  return (
    <div
      className="flex flex-col w-full cursor-pointer transition-all duration-300 hover:bg-foreground_alt bg-foreground p-5 rounded-xl flex-grow justify-start items-start gap-2 chat-animation"
      onClick={() => setTicketCollapsed((prev) => !prev)}
    >
      <div className="flex flex-col gap-2 w-full">
        <div className="flex justify-between items-start w-full">
          <div className="flex flex-col items-start gap-1">
            <p className="text-primary text-base font-bold">
              {ticket.issue_title}
            </p>
            <p className="text-xs font-light text-secondary">
              <span className="font-bold">{ticket.issue_author}</span> opened
              this on {formatDate(ticket.issue_created_at)}
            </p>
          </div>
          {ticket.issue_url && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                openLink();
              }}
              className="text-secondary btn btn-round"
            >
              <FaExternalLinkAlt size={12} />
            </button>
          )}
        </div>
        <div
          className={`text-primary overflow-scroll text-sm gap-5 mt-2 max-w-[70vw] flex flex-col text-wrap ${
            ticketCollapsed ? "max-h-[10vh]" : "max-h-[32vh]"
          }`}
        >
          <MarkdownMessageDisplay text={ticket.issue_content} />
        </div>
      </div>
    </div>
  );
};

export default TicketMessageDisplay;
