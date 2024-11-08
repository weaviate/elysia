"use client";

import React, { useState } from "react";
import { AiFillHome } from "react-icons/ai";
import { FaDatabase, FaExpandAlt } from "react-icons/fa";
import { LuRefreshCw } from "react-icons/lu";
import { IoMdAddCircle } from "react-icons/io";
import { IoChatbubble } from "react-icons/io5";
import { ImShrink2 } from "react-icons/im";

import { Collection, Conversation } from "../types";
import SidebarButton from "./sidebar_button";

interface SidebarProps {
  page: "home" | "data-explorer";
  handlePageChange: (p: "home" | "data-explorer") => void;
  addConversation: () => void;
  removeConversation: (id: string) => void;
  selectConversation: (id: string) => void;
  fetchCollections: () => void;
  selectCollection: (collection: string) => void;
  selectedCollection: string | null;
  currentConversation: string;
  conversations: Conversation[];
  socketOnline: boolean;
  collections: Collection[];
}

const Sidebar: React.FC<SidebarProps> = ({
  handlePageChange,
  page,
  addConversation,
  conversations,
  removeConversation,
  selectConversation,
  currentConversation,
  socketOnline,
  collections,
  fetchCollections,
  selectCollection,
  selectedCollection,
}) => {
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapse = () => {
    setCollapsed((prev) => !prev);
  };

  const asideClassName = `h-screen bg-background_alt p-5 flex flex-col gap-12 transition-transform duration-200 ${
    collapsed ? "w-[90px]" : "w-[18vw]"
  }`;

  const logoSize = collapsed ? "w-10 h-10" : "w-6 h-6";

  return (
    <aside className={asideClassName}>
      <div
        className={`flex items-center gap-1 ${
          collapsed ? "justify-center" : "justify-between"
        }`}
      >
        <div
          className={`flex items-center gap-1 ${
            collapsed ? "justify-center" : "justify-start"
          }`}
        >
          <img
            src={socketOnline ? "/logo.svg" : "/logo_offline.svg"}
            alt="elysia"
            className={`transition-all duration-200 ${logoSize}`}
          />
          {!collapsed && (
            <p className="text-sm font-extrabold text-primary">Elysia</p>
          )}
        </div>
        {!collapsed && (
          <button className="btn-round text-secondary" onClick={toggleCollapse}>
            <ImShrink2 size={14} />
          </button>
        )}
      </div>
      <nav className="flex flex-col h-full gap-6">
        <div className="flex flex-col gap-2">
          <SidebarButton
            icon={<AiFillHome />}
            label="Home"
            isActive={page === "home"}
            isCollapsed={collapsed}
            onClick={() => handlePageChange("home")}
          />
          <SidebarButton
            icon={<FaDatabase />}
            label="Data Explorer"
            isActive={page === "data-explorer"}
            isCollapsed={collapsed}
            onClick={() => handlePageChange("data-explorer")}
          />
        </div>
        <div className="border-t border-secondary"></div>
        {page === "home" && (
          <div className="flex flex-col gap-6">
            <SidebarButton
              icon={<IoMdAddCircle />}
              label="New Conversation"
              onClick={addConversation}
              isCollapsed={collapsed}
              onDelete={null}
            />
            <div className="flex flex-col gap-2">
              {conversations?.map((c) => (
                <SidebarButton
                  key={c.id}
                  icon={<IoChatbubble />}
                  label={c.name}
                  isCollapsed={collapsed}
                  isActive={currentConversation === c.id}
                  onClick={() => selectConversation(c.id)}
                  onDelete={() => removeConversation(c.id)}
                />
              ))}
            </div>
          </div>
        )}
        {page === "data-explorer" && (
          <div className="flex flex-col gap-6">
            <SidebarButton
              icon={<LuRefreshCw />}
              label="Refresh"
              onClick={fetchCollections}
              isCollapsed={collapsed}
              onDelete={null}
            />
            <div className="flex flex-col gap-2">
              {collections?.map((c) => (
                <SidebarButton
                  key={c.name}
                  icon={<FaDatabase />}
                  label={c.name}
                  isCollapsed={collapsed}
                  isActive={selectedCollection === c.name}
                  onClick={() => selectCollection(c.name)}
                />
              ))}
            </div>
          </div>
        )}
      </nav>
      {collapsed && (
        <div className="flex items-center justify-center w-full">
          <button className="btn-round text-secondary" onClick={toggleCollapse}>
            <FaExpandAlt size={20} />
          </button>
        </div>
      )}
    </aside>
  );
};

export default Sidebar;
