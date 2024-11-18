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
  const [pageSize, setPageSize] = useState(25);
  const [maxPage, setMaxPage] = useState(0);

  useEffect(() => {
    fetchCollections();
  }, []);

  useEffect(() => {
    if (!selectedCollection) return;
    fetchCollectionData();
  }, [selectedCollection, page]);

  useEffect(() => {
    const currentCollection = collections.find(
      (c) => c.name === selectedCollection
    );
    if (!currentCollection) return;
    setMaxPage(Math.ceil(currentCollection.total / pageSize) - 1);
  }, [collections, pageSize, selectedCollection]);

  const fetchCollections = async () => {
    setSelectedCollection(null);
    setCollections([]);
    setCollectionData(null);
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

  const selectCollection = async (collection: string | null) => {
    setSelectedCollection(collection);
    setPage(0);
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
    maxPage,
  };
}
