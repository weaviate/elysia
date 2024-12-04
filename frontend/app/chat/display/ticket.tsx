"use client";

import React, { useState } from "react";
import { Ticket } from "../../types";
import MarkdownMessageDisplay from "./markdown";
import { IoLink } from "react-icons/io5";

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
    window.open(ticket.url, "_blank");
  };

  const toggleSummary = () => {
    setShowSummary((prev) => !prev);
  };

  return (
    <div
      className="flex flex-col flex-grow w-full cursor-pointer transition-all duration-300 hover:bg-foreground_alt bg-foreground p-5 shadow-lg rounded-xl justify-start items-start gap-2 chat-animation"
      onClick={() => setTicketCollapsed((prev) => !prev)}
    >
      <div className="flex flex-col gap-2 w-full">
        <div className="flex items-start w-full">
          <div className="flex flex-col items-start gap-1 w-full">
            <div className="flex justify-between items-center gap-2 w-full">
              <div className="flex items-center gap-1">
                <p className="text-primary text-base font-bold">
                  {ticket.title}
                </p>
                {ticket.url && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      openLink();
                    }}
                    className="text-secondary btn btn-round"
                  >
                    <IoLink size={12} />
                  </button>
                )}
                <div className="flex items-center gap-2">
                  {ticket.status === "open" && (
                    <div className="p-2 bg-background_accent rounded-lg">
                      <p className="text-xs font-bold text-white">Open</p>
                    </div>
                  )}
                  {ticket.status === "closed" && (
                    <div className="p-2 bg-background_error rounded-lg">
                      <p className="text-xs font-bold text-white">Closed</p>
                    </div>
                  )}
                  {ticket.status !== "open" && ticket.status !== "closed" && (
                    <div className="p-2 bg-background_secondary rounded-lg">
                      <p className="text-xs font-bold text-white">
                        {ticket.status}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2">
                  {ticket.tags.length > 0 &&
                    ticket.tags.map((label, idx) => (
                      <div
                        key={`${idx}-${label}`}
                        className="p-2 bg-background_secondary rounded-lg"
                      >
                        <p className="text-xs font-bold text-white">{label}</p>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <p className="text-xs font-light text-secondary">
                <span className="font-bold">{ticket.author}</span> opened this
                on {formatDate(ticket.created_at)}
              </p>
              {ticket.summary && (
                <button
                  className="btn-static text-xs text-secondary"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleSummary();
                  }}
                >
                  |
                  <p>
                    {ticket.summary && showSummary
                      ? "Show Original"
                      : "Show Summary"}
                  </p>
                </button>
              )}
            </div>
          </div>
        </div>

        {((!ticketCollapsed && !ticket.summary) ||
          (!showSummary && ticket.summary)) && (
          <div className="text-primary overflow-scroll text-sm gap-5 mt-2 flex max-h-[50vh] flex-col text-wrap">
            <MarkdownMessageDisplay text={ticket.content} />
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
