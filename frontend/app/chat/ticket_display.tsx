"use client";

import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { MdError } from "react-icons/md";

import { Ticket } from "../types";
import { FaChevronDown } from "react-icons/fa";
import { TbDots } from "react-icons/tb";

interface TicketMessageDisplayProps {
  ticket: Ticket;
}

const TicketMessageDisplay: React.FC<TicketMessageDisplayProps> = ({
  ticket,
}) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div
      className="flex flex-col w-full cursor-pointer hover:bg-foreground_alt bg-foreground p-5 rounded-xl flex-grow justify-start items-start gap-2 chat-animation"
      onClick={() => setCollapsed(!collapsed)}
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
          className={`text-primary overflow-scroll text-sm gap-5 max-w-[65vw] flex flex-col text-wrap ${
            collapsed ? "max-h-[15vh]" : "max-h-[50vh]"
          }`}
        >
          <ReactMarkdown>{ticket.issue_content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default TicketMessageDisplay;
