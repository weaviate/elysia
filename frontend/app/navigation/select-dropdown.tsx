"use client";

import React, { useState, useEffect, useRef } from "react";
import { PiSelectionAllFill } from "react-icons/pi";
import { FaCheckCircle } from "react-icons/fa";
import { MdCancel } from "react-icons/md";

interface SelectDropdownProps {
  title: string;
  selections: { [key: string]: boolean };
  toggleOption: (option: string, conversationId: string) => void;
  currentConversation: string;
}

const SelectDropdown: React.FC<SelectDropdownProps> = ({
  title,
  selections,
  currentConversation,
  toggleOption,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  const selectAll = () => {
    Object.keys(selections).forEach((key) => {
      if (!selections[key]) {
        toggleOption(key, currentConversation);
      }
    });
  };

  const deselectAll = () => {
    Object.keys(selections).forEach((key) => {
      if (selections[key]) {
        toggleOption(key, currentConversation);
      }
    });
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="btn flex gap-2 bg-background_alt text-secondary text-xs"
      >
        <PiSelectionAllFill className="" />
        <p className="">{title}</p>
      </button>

      {isOpen && (
        <div className="absolute mt-5 flex flex-col gap-2 top-full bg-background_alt rounded-lg shadow-lg p-4 z-50">
          <div className="flex justify-between items-center gap-4">
            <button
              onClick={selectAll}
              className="btn-static text-xs text-secondary"
            >
              Select All
            </button>
            <button
              onClick={deselectAll}
              className="btn-static text-xs text-secondary"
            >
              Deselect All
            </button>
          </div>

          {Object.entries(selections).map(([key, value]) => (
            <div className="flex items-center justify-start gap-4" key={key}>
              <div
                onClick={() => {
                  toggleOption(key, currentConversation);
                }}
                className="btn cursor-pointer"
              >
                <div
                  className={`flex items-center justify-center ${
                    value ? "text-accent" : "text-error"
                  }`}
                >
                  {value ? <FaCheckCircle size={14} /> : <MdCancel size={14} />}
                </div>
                <p
                  className={`${
                    value ? "text-primary" : "text-secondary"
                  } text-xs`}
                >
                  {key}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SelectDropdown;
