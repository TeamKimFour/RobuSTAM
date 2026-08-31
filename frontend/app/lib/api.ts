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

// src/api/schemas.py ModelRun/DeployedModel과 필드명 그대로(snake_case) — 백엔드 응답을
// 변형 없이 받는다. validMdd·testSharpe는 백엔드가 아예 내려주지 않는 값이라 여기도 없다.
export type ModelRun = {
  run_id: string;
  fold_id: number;
  seed: number;
  total_timesteps: number;
  valid_sharpe: number | null;
  status: string;
  deployed: boolean;
};

export type DeployedModel = {
  run_id: string;
  model_version: string;
  model_path: string;
  scaler_fold_id: number;
  valid_sharpe: number | null;
  hyperparams: Record<string, string>;
};

export function fetchModelRuns(): Promise<ApiResult<ModelRun[]>> {
  return requestJson<ModelRun[]>("/models/runs");
}

export function fetchDeployedModel(): Promise<ApiResult<DeployedModel>> {
  return requestJson<DeployedModel>("/models/deployed");
}
