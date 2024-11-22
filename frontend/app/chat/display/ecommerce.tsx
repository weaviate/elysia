"use client";

import React from "react";
import { Ecommerce } from "../../types";
interface EcommerceDisplayProps {
  payload: Ecommerce[];
  routerChangeCollection: (collection_id: string) => void;
}

const EcommerceDisplay: React.FC<EcommerceDisplayProps> = ({
  payload,
  routerChangeCollection,
}) => {
  return (
    <div>
      {payload.map((product) => (
        <div key={product.name} className="flex flex-col">
          <p>{product.name}</p>
          <img src={product.image_url} alt={product.name} />
        </div>
      ))}
    </div>
  );
};

export default EcommerceDisplay;
