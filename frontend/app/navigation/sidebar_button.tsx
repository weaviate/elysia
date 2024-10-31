import React from "react";

interface SidebarButtonProps {
  icon: React.ReactNode;
  label: string;
  isActive?: boolean;
  isCollapsed: boolean;
  onClick?: () => void;
}

const SidebarButton: React.FC<SidebarButtonProps> = ({
  icon,
  label,
  isActive,
  isCollapsed,
  onClick,
}) => (
  <button
    className={`btn ${
      isActive ? "bg-foreground text-primary" : "text-secondary"
    } ${isCollapsed ? "justify-center" : "justify-start"}`}
    onClick={onClick}
  >
    {React.cloneElement(icon as React.ReactElement, {
      size: isCollapsed ? 20 : 14,
    })}
    {!isCollapsed && <p className="text-xs font-medium">{label}</p>}
  </button>
);

export default SidebarButton;
