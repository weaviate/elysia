import { useEffect, useState } from "react";
import { Collection, CollectionData } from "../types";
import { getCollection, getCollections } from "./api";
export function useCollections() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(false);

  const [selectedCollection, setSelectedCollection] = useState<string | null>(
    null
  );

  const [collectionData, setCollectionData] = useState<CollectionData | null>(
    null
  );

  const [loadingCollection, setLoadingCollection] = useState(false);

  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(100);

  useEffect(() => {
    fetchCollections();
  }, []);

  useEffect(() => {
    if (!selectedCollection) return;
    fetchCollectionData();
  }, [selectedCollection]);

  const fetchCollections = async () => {
    setLoadingCollections(true);
    const collections = await getCollections();
    setCollections(collections);
    setLoadingCollections(false);
  };

  const fetchCollectionData = async () => {
    if (!selectedCollection) return;
    setLoadingCollection(true);
    const collectionData = await getCollection(
      selectedCollection,
      page,
      pageSize
    );
    setCollectionData(collectionData);
    setLoadingCollection(false);
  };

  const selectCollection = async (collection: string) => {
    setSelectedCollection(collection);
  };

  const pageUp = () => {
    if (!selectedCollection) return;
    const collection = collections.find((c) => c.name === selectedCollection);
    if (!collection) return;
    if ((page + 1) * pageSize >= collection.total) return;
    setPage((prev) => prev + 1);
  };

  const pageDown = () => {
    if (page === 0) return;
    setPage((prev) => prev - 1);
  };

  return {
    collections,
    fetchCollections,
    selectCollection,
    selectedCollection,
    loadingCollections,
    loadingCollection,
    collectionData,
    page,
    setPage,
    pageSize,
    setPageSize,
    pageUp,
    pageDown,
  };
}
