import { useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Collection } from "../types";

export function useRouting(
  handleModeChange: (mode: "home" | "data-explorer") => void,
  currentMode: "home" | "data-explorer",
  handleCollectionSelect: (collection_id: string | null) => void,
  selectedCollection: string | null,
  collections: Collection[],
  maxPage: number,
  page: number,
  setPage: React.Dispatch<React.SetStateAction<number>>
) {
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const mode = searchParams.get("mode");
    const collection_id = searchParams.get("collection_id");
    if (mode !== currentMode && (mode === "data-explorer" || mode === "home")) {
      handleModeChange(mode);
    }
    if (collection_id) {
      handleCollectionSelect(collection_id);
    } else {
      handleCollectionSelect(null);
    }

    const page = searchParams.get("page");
    if (page) {
      setPage(parseInt(page) - 1);
    }
  }, [searchParams]);

  const routerToLogin = () => {
    router.push("/login");
  };

  const routerChangeMode = (mode: "home" | "data-explorer") => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("mode", mode);
    if (mode === "home") {
      params.delete("collection_id");
      params.delete("page");
    } else if (mode === "data-explorer") {
      params.delete("collection_id");
      params.delete("page");
    }
    router.push(`/?${params.toString()}`);
  };

  const routerChangeCollection = (collection_id: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("collection_id", collection_id);
    params.set("page", "1");
    params.set("mode", "data-explorer");
    router.push(`/?${params.toString()}`);
  };

  const routerSetPage = (page: number) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("page", (page + 1).toString());
    router.push(`/?${params.toString()}`);
  };

  const pageUp = () => {
    if (!selectedCollection) return;
    const collection = collections.find((c) => c.name === selectedCollection);
    if (!collection) return;
    if (page + 1 > maxPage) return;
    routerSetPage(page + 1);
  };

  const pageUpMax = () => {
    routerSetPage(maxPage);
  };

  const pageDown = () => {
    if (page === 0) return;
    routerSetPage(page - 1);
  };

  const pageDownMax = () => {
    routerSetPage(0);
  };

  return {
    routerChangeMode,
    routerChangeCollection,
    routerToLogin,
    pageUp,
    pageUpMax,
    pageDown,
    pageDownMax,
  };
}
