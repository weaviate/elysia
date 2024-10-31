"use client";

import React from "react";
import { AiFillHome } from "react-icons/ai";
import { FaDatabase } from "react-icons/fa";
import { FaTicketSimple } from "react-icons/fa6";
import { IoMdAddCircle } from "react-icons/io";

interface SidebarProps {
  page: "home" | "data-explorer";
  handlePageChange: (p: "home" | "data-explorer") => void;
}

const Sidebar: React.FC<SidebarProps> = ({ handlePageChange, page }) => {
  return (
    <aside className="h-screen bg-background_alt p-5 flex flex-col gap-12">
      <div className="flex items-center justify-start gap-1">
        <img src="/logo.svg" alt="elysia" className="w-6 h-6" />
        <p className="text-sm font-extrabold text-primary">Elysia</p>
      </div>
      <nav className="flex flex-col h-full gap-6">
        <div className="flex flex-col gap-2">
          <button
            className={`btn ${
              page === "home" ? "bg-foreground text-primary" : "text-secondary"
            }`}
            onClick={() => handlePageChange("home")}
          >
            <AiFillHome size={14} />
            <p className="text-xs font-medium ">Home</p>
          </button>
          <button
            className={`btn ${
              page === "data-explorer"
                ? "bg-foreground text-primary"
                : "text-secondary"
            }`}
            onClick={() => handlePageChange("data-explorer")}
          >
            <FaDatabase size={14} />
            <p className="text-xs font-medium ">Data Explorer</p>
          </button>
        </div>
        <div className="border-t border-secondary my-2"></div>
        <div className="flex flex-col gap-2">
          <button className={`btn text-secondary`}>
            <IoMdAddCircle size={14} />
            <p className="text-xs font-medium ">New Chat</p>
          </button>
          <button className={`btn bg-foreground text-primary`}>
            <FaTicketSimple size={14} />
            <p className="text-xs font-medium ">Current Chat</p>
          </button>
        </div>
      </nav>
    </aside>
  );
};

export default Sidebar;
