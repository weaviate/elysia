import React from "react";
import { FaMinusCircle } from "react-icons/fa";

interface SidebarButtonProps {
  icon: React.ReactNode;
  label: string;
  isActive?: boolean;
  isCollapsed: boolean;
  onClick?: () => void;
  onDelete?: (() => void) | null;
}

const SidebarButton: React.FC<SidebarButtonProps> = ({
  icon,
  label,
  isActive,
  isCollapsed,
  onClick,
  onDelete,
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
    {!isCollapsed && (
      <div className="flex w-full items-center justify-between gap-2">
        <p className="text-xs font-medium">{label}</p>
        {onDelete && (
          <div
            className="btn-round cursor-pointer text-secondary"
            onClick={onDelete}
          >
            <FaMinusCircle size={12} />
          </div>
        )}
      </div>
    )}
  </button>
);

export default SidebarButton;
