"use client";

import React from "react";

interface DashboardProps {}

const Dashboard: React.FC<DashboardProps> = ({}) => {
  return (
    <div
      className="flex w-[80vw] flex-col p-8 h-screen items-start justify-start outline-none"
      tabIndex={0}
    >
      <p className="text-xl font-bold font-merriweather text-primary">
        Data Dashboard
      </p>
    </div>
  );
};

export default Dashboard;
