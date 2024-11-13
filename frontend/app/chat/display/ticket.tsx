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
  const [showSummary, setShowSummary] = useState(ticket.summary ? true : false);

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

  const toggleSummary = () => {
    setShowSummary((prev) => !prev);
  };

  return (
    <div
      className="flex flex-col flex-grow w-full cursor-pointer transition-all duration-300 hover:bg-foreground_alt bg-background_alt p-5 rounded-xl justify-start items-start gap-2 chat-animation"
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
          <div className="flex gap-2">
            {ticket.summary && (
              <button
                className="btn-static text-xs text-secondary"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleSummary();
                }}
              >
                <p>
                  {ticket.summary && showSummary
                    ? "Show Original"
                    : "Show Summary"}
                </p>
              </button>
            )}
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
        </div>
        {((!ticketCollapsed && !ticket.summary) ||
          (!showSummary && ticket.summary)) && (
          <div className="text-primary overflow-scroll text-sm gap-5 mt-2 flex max-h-[50vh] flex-col text-wrap">
            <MarkdownMessageDisplay text={ticket.issue_content} />
          </div>
        )}
        {showSummary && ticket.summary && (
          <div className="text-primary overflow-scroll text-sm gap-5 mt-2 flex flex-col text-wrap">
            <MarkdownMessageDisplay text={ticket.summary} />
          </div>
        )}
      </div>
    </div>
  );
};

export default TicketMessageDisplay;
