/**
 * Thin wrapper over `fetch` for talking to the FastAPI backend.
 *
 * All requests go through `/api/*` so they're same-origin during dev
 * (next.config.ts rewrites them to the FastAPI port).
 */

import type { HealthResponse, PredictionResponse } from "@/lib/types";

const API_BASE = "/api";

export class APIError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "APIError";
  }
}

export async function health(): Promise<HealthResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  } catch {
    throw new APIError(
      0,
      "Cannot reach the API — is the FastAPI backend running on port 8000?",
    );
  }
  if (!res.ok) {
    throw new APIError(
      res.status,
      `Health check failed: ${res.status} ${res.statusText}`,
    );
  }
  return res.json();
}

export async function predict(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  } catch {
    throw new APIError(
      0,
      "Cannot reach the API — please ensure the backend is running on port 8000.",
    );
  }
  if (!res.ok) {
    let detail: string = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // body is not JSON — keep statusText
    }
    throw new APIError(res.status, detail);
  }
  return res.json();
}
