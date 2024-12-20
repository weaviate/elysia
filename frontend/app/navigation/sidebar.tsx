"use client";

import React, { useState } from "react";
import { AiFillHome } from "react-icons/ai";
import { FaCircle, FaDatabase, FaExpandAlt } from "react-icons/fa";
import { LuRefreshCw } from "react-icons/lu";
import { IoMdAddCircle } from "react-icons/io";
import { IoChatbubble } from "react-icons/io5";
import { ImShrink2 } from "react-icons/im";

import { Collection, Conversation } from "../types";
import SidebarButton from "@/app/navigation/sidebar-button";

interface SidebarProps {
  mode: "home" | "data-explorer";
  handleModeChange: (p: "home" | "data-explorer") => void;
  addConversation: () => void;
  removeConversation: (id: string) => void;
  selectConversation: (id: string) => void;
  fetchCollections: () => void;
  selectCollection: (collection: string) => void;
  selectedCollection: string | null;
  currentConversation: string;
  conversations: Conversation[];
  socketOnline: boolean;
  routerToLogin: () => void;
  collections: Collection[];
  creatingNewConversation: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  handleModeChange,
  mode,
  addConversation,
  conversations,
  removeConversation,
  routerToLogin,
  selectConversation,
  currentConversation,
  socketOnline,
  collections,
  fetchCollections,
  selectCollection,
  selectedCollection,
  creatingNewConversation,
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
          onClick={routerToLogin}
          className={`flex items-center gap-1 cursor-pointer ${
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
            isActive={mode === "home"}
            isCollapsed={collapsed}
            onClick={() => handleModeChange("home")}
          />
          <SidebarButton
            icon={<FaDatabase />}
            label="Data Explorer"
            isActive={mode === "data-explorer"}
            isCollapsed={collapsed}
            onClick={() => handleModeChange("data-explorer")}
          />
        </div>
        <div className="border-t border-secondary"></div>
        {mode === "home" && (
          <div className="flex flex-col gap-6">
            {!creatingNewConversation ? (
              <SidebarButton
                icon={<IoMdAddCircle />}
                label="New Conversation"
                onClick={addConversation}
                isCollapsed={collapsed}
                onDelete={null}
              />
            ) : (
              <button className="btn bg-background_alt flex items-center justify-start gap-2">
                <FaCircle className="text-secondary text-xs pulsing" />
                <p className="text-xs font-medium truncate max-w-[10vw] shine">
                  Creating new conversation...
                </p>
              </button>
            )}
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
        {mode === "data-explorer" && (
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
