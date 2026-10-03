/**
 * JSON shapes returned by the Traffilytics API.
 *
 * Mirror backend payloads; keep fields optional when the worker may omit
 * them. `Units` is the provenance for speed/density — pixel vs ground scale.
 */
export type Units = {
  speed?: string;
  density?: string;
  pixels_per_metre?: number | null;
  labelled_as_physical?: boolean;
  note?: string;
};

export type VideoRow = {
  video_id: string;
  site?: string | null;
  location?: string | null;
  duration?: number | null;
  fps?: number | null;
  resolution?: string | null;
  source?: string | null;
  stabilized?: boolean;
  status?: string;
  scale_m_per_px?: number | null;
  lane_config?: Record<string, unknown> | null;
  model_version?: string | null;
  tracker_name?: string | null;
  created_at?: string | null;
};

export type JobRow = {
  job_id: string;
  video_id: string;
  stage: string;
  progress: number;
  status: string;
  error?: string | null;
  kind?: string;
  updated_at?: string;
};

export type UploadResponse = {
  video_id: string;
  job_id: string;
  site?: string | null;
  fps?: number | null;
  resolution?: string | null;
  duration?: number | null;
  source?: string;
  status: string;
};

export type Overview = VideoRow & {
  n_vehicles?: number;
  traffic_state?: string;
  mean_speed?: number | null;
  vehicles_per_minute?: number | null;
  units?: Units;
  major_events?: Array<Record<string, unknown>>;
  n_events?: number;
  insights?: Array<{ id?: string; text?: string } | string>;
};

export type FlowWindow = {
  t0: number;
  t1: number;
  n_vehicles: number;
  volume_per_min: number;
  mean_speed: number | null;
  density: number | null;
  occupancy_mean: number;
  flow: number | null;
  state: string;
};

export type FlowPayload = {
  video_id: string;
  units?: Units;
  flow: {
    vehicles_per_minute?: number;
    mean_speed?: number | null;
    mean_density?: number | null;
    free_flow_speed?: number | null;
    windows?: FlowWindow[];
    flow_density?: Array<{
      density: number | null;
      flow: number | null;
      mean_speed: number | null;
      state: string;
      t0: number;
      t1: number;
    }>;
  };
};

export type EventRow = {
  event_id?: string;
  event_type?: string;
  type?: string;
  timestamp?: string;
  time?: string;
  severity?: string;
  location?: string;
  zone_id?: string;
  track_id?: number | null;
};

export type LaneShare = {
  lane: string;
  n_tracks: number;
  share: number;
  mean_heading?: number | null;
  mean_speed?: number | null;
};

export type ReportPayload = {
  title?: string;
  generated_at?: string;
  site?: string | null;
  units?: Units;
  overview?: {
    n_vehicles?: number;
    traffic_state?: string;
    mean_speed?: number | null;
    n_events?: number;
  };
  findings?: string[];
  attribution?: {
    uav_obb?: string;
    licence?: string;
    source_commit?: string;
    source_commit_url?: string;
    release_tag?: string;
    release_url?: string;
    weights_url?: string;
  };
};

export type VehicleRow = {
  video_id: string;
  track_id: number;
  class_id: number;
  vehicle_type?: string | null;
  entry_frame?: number | null;
  exit_frame?: number | null;
  entry_time?: string | null;
  exit_time?: string | null;
};

export type MapPoint = {
  x: number;
  y: number;
  frame: number;
  speed?: number | null;
};

export type MapTrack = {
  track_id: number;
  class_id?: number | null;
  vehicle_type?: string | null;
  n_points?: number;
  points: MapPoint[];
};

export type MapPayload = {
  video_id: string;
  stride: number;
  n_tracks: number;
  n_points: number;
  bounds?: { x0: number; y0: number; x1: number; y1: number } | null;
  tracks: MapTrack[];
};

export type FramePreview = {
  video_id: string;
  n_frames: number;
  default_i: number;
  frame: number;
  indices?: number[];
  width: number | null;
  height: number | null;
  overlay?: boolean;
};

export type HeatCell = {
  row: number;
  col: number;
  n_points: number;
  mean_speed: number | null;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
};

export type BottleneckPayload = {
  units?: Units;
  bottleneck: {
    configured: boolean;
    primary?: { zone?: string; cause?: string } | null;
    zones?: Array<Record<string, unknown>>;
    note?: string;
  };
};

export type ImbalancePayload = {
  units?: Units;
  imbalance: {
    configured: boolean;
    lanes?: LaneShare[];
    imbalance_index?: number | null;
    dominant_lane?: string | null;
    note?: string;
  };
};
