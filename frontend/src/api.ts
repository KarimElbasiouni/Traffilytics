import type {
  BottleneckPayload,
  EventRow,
  FlowPayload,
  HeatCell,
  ImbalancePayload,
  JobRow,
  MapPayload,
  Overview,
  ReportPayload,
  FramePreview,
  UploadResponse,
  VehicleRow,
  VideoRow,
} from "./types";

const API = "/api/v1";

export class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  const data = (await res.json().catch(() => ({}))) as {
    error?: { code?: string; message?: string };
  };
  if (!res.ok) {
    throw new ApiError(
      data.error?.code || "error",
      data.error?.message || res.statusText,
    );
  }
  return data as T;
}

export const api = {
  videos: () => request<{ videos: VideoRow[] }>("/videos"),
  video: (id: string) => request<VideoRow>(`/videos/${id}`),
  upload: (body: FormData) =>
    request<UploadResponse>("/videos", { method: "POST", body }),
  process: (id: string) =>
    request<{ video_id: string; job_id: string; status: string }>(
      `/videos/${id}/process`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reuse_artifacts: true }),
      },
    ),
  job: (id: string) => request<JobRow>(`/jobs/${id}`),
  overview: (id: string) => request<Overview>(`/videos/${id}/overview`),
  flow: (id: string) => request<FlowPayload>(`/videos/${id}/analytics/flow`),
  flowDensity: (id: string) =>
    request<{ flow_density: FlowPayload["flow"]["flow_density"]; units?: Overview["units"] }>(
      `/videos/${id}/analytics/flow-density`,
    ),
  bottlenecks: (id: string) =>
    request<BottleneckPayload>(`/videos/${id}/analytics/bottlenecks`),
  heatmap: (id: string) =>
    request<{ heatmap: HeatCell[] }>(`/videos/${id}/analytics/heatmap`),
  imbalance: (id: string) =>
    request<ImbalancePayload>(`/videos/${id}/analytics/imbalance`),
  putLanes: (id: string, body: unknown) =>
    request<{ path?: string }>(`/videos/${id}/lanes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  events: (id: string) =>
    request<{ events: EventRow[]; n_events: number }>(`/videos/${id}/events`),
  reports: (id: string) => request<ReportPayload>(`/videos/${id}/reports`),
  vehicles: (id: string) =>
    request<{ vehicles: VehicleRow[]; n_vehicles: number }>(`/videos/${id}/vehicles`),
  preview: (id: string) => request<FramePreview>(`/videos/${id}/preview`),
  frameUrl: (id: string, i?: number) =>
    i == null
      ? `${API}/videos/${id}/frame`
      : `${API}/videos/${id}/frame?i=${encodeURIComponent(String(i))}`,
  overlayUrl: (id: string) => `${API}/videos/${id}/overlay?v=boxes`,
  map: (id: string) => request<MapPayload>(`/videos/${id}/map?stride=5`),
};
