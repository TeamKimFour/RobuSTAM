export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export type LatestInference = {
  date: string;
  generated_at: string;
  model_version: string;
  weights: Record<string, number>;
};

export type ApiFailure =
  | { kind: "network"; message: string }
  | { kind: "notReady"; status: number; message: string }
  | { kind: "http"; status: number; message: string };

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ApiFailure };

async function requestJson<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const url = `${API_BASE_URL}${path}`;
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store", ...init });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return { ok: false, error: { kind: "network", message } };
  }

  if (res.ok) {
    const data = (await res.json()) as T;
    return { ok: true, data };
  }

  let detail = "";
  try {
    const body = (await res.json()) as { detail?: string };
    detail = body.detail ?? "";
  } catch {
    detail = await res.text().catch(() => "");
  }

  if (res.status === 503) {
    return {
      ok: false,
      error: { kind: "notReady", status: 503, message: detail || "precompute 미준비" },
    };
  }
  return {
    ok: false,
    error: { kind: "http", status: res.status, message: detail || res.statusText },
  };
}

export function fetchLatestInference(): Promise<ApiResult<LatestInference>> {
  return requestJson<LatestInference>("/inference/latest");
}

export function fetchHealth(): Promise<ApiResult<{ status: string }>> {
  return requestJson<{ status: string }>("/health");
}
