"use client";

import React from "react";
import { Ecommerce, ResultPayload } from "../../types";

interface EcommerceDisplayProps {
  payload: ResultPayload;
}

const EcommerceDisplay: React.FC<EcommerceDisplayProps> = ({ payload }) => {
  const products = payload.objects as Ecommerce[];
  return (
    <div className="w-full flex flex-col gap-4">
      <div className="w-full flex gap-4 overflow-x-auto pb-4">
        {products.map((product, idx) => (
          <div
            key={`${idx}-${product.name}`}
            className="flex flex-col gap-2 w-[20vw] flex-shrink-0 bg-background_alt p-4 rounded-lg shadow-xl"
          >
            <div className="flex flex-col gap-2 items-start justify-start">
              <p className="text-xs text-secondary font-light">
                {product.collection} {" | "} {product.category} {" | "}
                {product.subcategory}
              </p>
              <img
                src={product.image_url}
                alt={product.name}
                className="rounded-lg"
              />
            </div>
            <div className="flex flex-col gap-3 items-start justify-start">
              <div className="flex flex-col overflow-scroll">
                <p className="text-xs text-secondary font-light">
                  {product.brand}
                </p>
                <p className="text-sm text-primary font-bold">{product.name}</p>
                <div className="flex items-center gap-1 w-full justify-start">
                  {[...Array(5)].map((_, i) => (
                    <span
                      key={i}
                      className={`text-xs ${
                        i < Math.round(product.rating)
                          ? "text-accent"
                          : "text-secondary"
                      }`}
                    >
                      ★
                    </span>
                  ))}
                </div>
              </div>
              <p className="text-xs text-wrap h-[5rem] overflow-y-auto">
                {product.description}
              </p>
              <div className="w-full flex justify-end">
                <p className="text-lg text-primary font-bold">
                  ${product.price}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EcommerceDisplay;
