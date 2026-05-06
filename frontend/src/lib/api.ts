/**
 * Thin fetch helpers. Same-origin by default; if NEXT_PUBLIC_API_BASE_URL is
 * set we prefix all calls with it (useful in dev when frontend and backend
 * are on different ports).
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? '';

function url(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  return BASE + path;
}

export async function fetcher<T = unknown>(path: string): Promise<T> {
  const res = await fetch(url(path), { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ''}`);
  }
  return res.json() as Promise<T>;
}

export async function api<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url(path), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ''}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export async function uploadFile<T = unknown>(path: string, file: File): Promise<T> {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(url(path), { method: 'POST', body: fd });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ''}`);
  }
  return res.json() as Promise<T>;
}

export function sseUrl(path: string): string {
  // If we're in the browser and using the proxy (BASE is empty), try to go 
  // direct to the backend port for SSE to avoid Next.js rewrite buffering.
  if (typeof window !== 'undefined' && !BASE && window.location.hostname === 'localhost') {
    return `http://localhost:8000${path}`;
  }
  return url(path);
}
