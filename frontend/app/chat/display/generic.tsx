"use client";

import DataTable from "@/app/explorer/table";

interface BoringGenericDisplayProps {
  payload: { [key: string]: string }[];
}

const BoringGenericDisplay: React.FC<BoringGenericDisplayProps> = ({
  payload,
}) => {
  return (
    <div className="w-full flex flex-col justify-start items-start mb-4">
      <DataTable
        data={payload}
        header={Object.keys(payload[0])}
        loading={false}
        setSelectedCell={() => {}}
      />
    </div>
  );
};

export default BoringGenericDisplay;
