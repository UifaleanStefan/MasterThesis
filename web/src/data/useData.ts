/**
 * Generic data-fetching hook. The site is fully static; data lives at
 * /data/<name>.json and is fetched on first mount.
 *
 * Usage:
 *   const { data, error, loading } = useData<BenchmarkData>("benchmark.json");
 */

import { useEffect, useState } from "react";

export interface DataState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

const _cache = new Map<string, unknown>();
const _inflight = new Map<string, Promise<unknown>>();

function _resolveUrl(name: string): string {
  // The site uses Vite's `base: './'` (relative paths). Build a relative
  // URL into the public/ folder.
  return `${import.meta.env.BASE_URL}data/${name}`.replace(/\/+/g, "/");
}

async function _fetchJson<T>(name: string): Promise<T> {
  if (_cache.has(name)) return _cache.get(name) as T;
  if (_inflight.has(name)) return _inflight.get(name) as Promise<T>;
  const url = _resolveUrl(name);
  const promise = fetch(url).then(async (res) => {
    if (!res.ok) {
      throw new Error(`Failed to load /data/${name} (${res.status} ${res.statusText})`);
    }
    const json = (await res.json()) as T;
    _cache.set(name, json);
    _inflight.delete(name);
    return json;
  });
  _inflight.set(name, promise);
  return promise;
}

export function useData<T>(name: string): DataState<T> {
  const [state, setState] = useState<DataState<T>>({
    data: (_cache.get(name) as T) ?? null,
    error: null,
    loading: !_cache.has(name),
  });

  useEffect(() => {
    let cancelled = false;
    if (_cache.has(name)) {
      setState({ data: _cache.get(name) as T, error: null, loading: false });
      return;
    }
    setState((s) => ({ ...s, loading: true }));
    _fetchJson<T>(name)
      .then((data) => {
        if (!cancelled) setState({ data, error: null, loading: false });
      })
      .catch((error: Error) => {
        if (!cancelled) setState({ data: null, error, loading: false });
      });
    return () => {
      cancelled = true;
    };
  }, [name]);

  return state;
}
