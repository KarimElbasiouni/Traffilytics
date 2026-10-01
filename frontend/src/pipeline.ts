/**
 * Ordered worker stages for a clip job.
 *
 * Keep in sync with backend job `stage` strings. The UI uses this list for
 * the progress rail and log; it does not run the pipeline itself.
 */
export const PIPELINE_STAGES = [
  "ingest",
  "detection",
  "tracking",
  "analytics",
  "persist",
  "completed",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];
