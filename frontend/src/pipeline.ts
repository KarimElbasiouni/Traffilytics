export const PIPELINE_STAGES = [
  "ingest",
  "detection",
  "tracking",
  "analytics",
  "persist",
  "completed",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];
