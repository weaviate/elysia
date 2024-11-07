"use client";

import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { MdError } from "react-icons/md";

import { TicketPayload } from "../types";

interface TicketMessageDisplayProps {
  ticket: TicketPayload;
}

const TicketMessageDisplay: React.FC<TicketMessageDisplayProps> = ({
  ticket,
}) => {
  return (
    <div className="flex bg-foreground p-6 rounded-xl flex-grow justify-start items-center gap-2 chat-animation">
      <div className="flex flex-col gap-2">
        <div className="flex flex-col items-start gap-1">
          <p className="text-xs font-light text-secondary">
            {ticket.issue_created_at}
          </p>
          <p className="text-primary text-base font-bold">
            {ticket.issue_title}
          </p>
        </div>
        <p className="text-primary text-sm">{ticket.issue_content}</p>
      </div>
    </div>
  );
};

export default TicketMessageDisplay;
