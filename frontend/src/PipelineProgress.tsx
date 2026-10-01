/**
 * Modal that follows a processing job through `PIPELINE_STAGES`.
 *
 * Fill is interpolated from the worker `stage` (not a fake 0–100 from the
 * API). `PipelineRail` is the stage dots; the list below is the same stages
 * with timestamps. Escape or backdrop click closes.
 */
import { useEffect, useRef, useState } from "react";
import type { JobRow } from "./types";
import { PIPELINE_STAGES, type PipelineStage } from "./pipeline";

const STAGE_COPY: Record<PipelineStage, { title: string; detail: string }> = {
  ingest: {
    title: "Ingest",
    detail: "Read the file and extract frames for the detector.",
  },
  detection: {
    title: "Detection",
    detail: "Find vehicles on each ingested frame.",
  },
  tracking: {
    title: "Tracking",
    detail: "Connect detections into tracks across frames.",
  },
  analytics: {
    title: "Analytics",
    detail: "Compute flow, occupancy, events, and findings.",
  },
  persist: {
    title: "Persist",
    detail: "Save results so Overview can read them.",
  },
  completed: {
    title: "Completed",
    detail: "The clip is ready on Overview.",
  },
};

function asStage(value?: string | null): PipelineStage | "queued" | "failed" {
  if (value === "failed") return "failed";
  if (value && (PIPELINE_STAGES as readonly string[]).includes(value)) {
    return value as PipelineStage;
  }
  return "queued";
}

function stageIndex(stage: PipelineStage | "queued" | "failed", lastGood: PipelineStage) {
  if (stage === "queued") return 0;
  if (stage === "failed") return PIPELINE_STAGES.indexOf(lastGood);
  return PIPELINE_STAGES.indexOf(stage);
}

function clock() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function usePipelineFill(stageIdx: number, status?: string) {
  const gaps = PIPELINE_STAGES.length - 1;
  const [creep, setCreep] = useState(0);
  const [fill, setFill] = useState(0);
  const started = useRef(performance.now());
  const targetRef = useRef(0);
  const reduce =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const target =
    status === "completed"
      ? 100
      : Math.min(
          98,
          (Math.max(0, stageIdx) / gaps + (status === "failed" ? 0 : (creep * 0.72) / gaps)) * 100,
        );
  targetRef.current = target;

  useEffect(() => {
    started.current = performance.now();
    setCreep(0);
    if (status === "completed") {
      setFill(100);
      return;
    }
    if (stageIdx === 0) setFill(0);
  }, [stageIdx, status]);

  useEffect(() => {
    if (status === "completed" || status === "failed" || reduce) {
      setCreep(0);
      return;
    }
    let frame = 0;
    const tick = (now: number) => {
      const t = (now - started.current) / 10000;
      setCreep(Math.min(0.72, 1 - Math.exp(-2.2 * t)));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [stageIdx, status, reduce]);

  useEffect(() => {
    if (status === "completed") {
      setFill(100);
      return;
    }
    if (reduce) {
      setFill(targetRef.current);
      return;
    }
    let frame = 0;
    const tick = () => {
      setFill((prev) => {
        const t = targetRef.current;
        const next = prev + (t - prev) * 0.12;
        return Math.abs(t - next) < 0.08 ? t : next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [status, reduce]);

  if (status === "completed") return 100;
  return reduce ? target : fill;
}

function nodeMark(
  i: number,
  current: number,
  status?: string,
  name?: PipelineStage,
): "done" | "now" | "wait" | "fail" {
  const failed = status === "failed";
  if (failed && i === current) return "fail";
  if (!failed && (status === "completed" || i < current || (i === current && name === "completed"))) {
    return "done";
  }
  if (i === current && status !== "completed") return "now";
  return "wait";
}

export function PipelineRail({
  stage,
  status,
  lastGood,
  fill,
}: {
  stage: PipelineStage | "queued" | "failed";
  status?: string;
  lastGood: PipelineStage;
  fill: number;
}) {
  const current = stageIndex(stage, lastGood);
  const x0 = 100 / 12;
  const x1 = 100 - x0;
  const stretch = status === "completed" ? 100 : Math.min(100, Math.max(0, fill));
  const xFill = x0 + ((x1 - x0) * stretch) / 100;
  return (
    <div className="pipe" role="group" aria-label="Processing stages">
      <div className="pipe-row">
        <svg className="pipe-svg" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">
          <line x1={x0} y1="4" x2={x1} y2="4" vectorEffect="nonScalingStroke" />
          <line className="pipe-svg-fill" x1={x0} y1="4" x2={xFill} y2="4" vectorEffect="nonScalingStroke" />
        </svg>
        <ol className="pipe-nodes">
          {PIPELINE_STAGES.map((name, i) => {
            const mark = nodeMark(i, current, status, name);
            return (
              <li key={name} className="pipe-node" data-mark={mark}>
                <span className="pipe-dot" aria-hidden="true">
                  {mark === "done" ? <CheckIcon /> : mark === "fail" ? <FailIcon /> : mark === "now" ? (
                    <span className="pipe-now-core" />
                  ) : null}
                </span>
                <span className="pipe-name">{STAGE_COPY[name].title}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

export function PipelineProgress({
  job,
  uploading,
  onClose,
  onOpenClip,
}: {
  job?: JobRow;
  uploading?: boolean;
  onClose: () => void;
  onOpenClip?: () => void;
}) {
  const [seen, setSeen] = useState<Partial<Record<PipelineStage, string>>>({});
  const lastGood = useRef<PipelineStage>("ingest");
  const raw = uploading ? "ingest" : job?.stage;
  const stage = asStage(raw);
  useEffect(() => {
    if (stage !== "queued" && stage !== "failed") lastGood.current = stage;
  }, [stage]);
  const idx = stageIndex(stage, lastGood.current);
  const status = uploading ? "processing" : job?.status;
  const fill = usePipelineFill(idx, status);
  const failed = status === "failed";
  const currentName = PIPELINE_STAGES[idx];
  const currentCopy = STAGE_COPY[currentName];

  useEffect(() => {
    setSeen(uploading ? { ingest: clock() } : {});
    lastGood.current = "ingest";
  }, [job?.job_id, uploading]);

  useEffect(() => {
    if (uploading) return;
    const name = asStage(job?.stage);
    if (name === "queued" || name === "failed") return;
    const until =
      job?.status === "completed" ? PIPELINE_STAGES.length - 1 : PIPELINE_STAGES.indexOf(name);
    setSeen((prev) => {
      const next = { ...prev };
      let changed = false;
      for (let i = 0; i <= until; i += 1) {
        const key = PIPELINE_STAGES[i];
        if (!next[key]) {
          next[key] = clock();
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [job?.stage, job?.status, uploading]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const heading = failed
    ? "Processing stopped"
    : status === "completed"
      ? "Processing finished"
      : "Processing your video";

  return (
    <div className="pipe-scrim" role="presentation" onClick={onClose}>
      <section
        className="pipe-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pipe-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="pipe-head">
          <span className="pipe-badge" aria-hidden="true">
            <UploadGlyph />
          </span>
          <div>
            <h2 id="pipe-title">{heading}</h2>
            <p>
              {failed
                ? job?.error || "The worker reported a failure."
                : status === "completed"
                  ? "Frames, tracks, and analytics are stored for this clip."
                  : "The file is on the server. Stages run in a worker, not in this tab."}
            </p>
          </div>
          <button type="button" className="pipe-x" onClick={onClose} aria-label="Close progress">
            ×
          </button>
        </header>

        <PipelineRail stage={stage} status={status} lastGood={lastGood.current} fill={fill} />

        <article
          className="pipe-current"
          data-state={failed ? "failed" : status === "completed" ? "done" : "live"}
        >
          <header>
            <span className="pipe-live-dot" aria-hidden="true" />
            <div>
              <h3>{currentCopy.title}</h3>
              <p>{failed ? job?.error || currentCopy.detail : currentCopy.detail}</p>
            </div>
            <b>{status === "completed" ? "100%" : `${Math.round(fill)}%`}</b>
          </header>
          <div className="pipe-meter" aria-hidden="true">
            <span style={{ width: `${status === "completed" ? 100 : fill}%` }} />
          </div>
        </article>

        <ul className="pipe-log">
          {PIPELINE_STAGES.map((name, i) => {
            const mark = nodeMark(i, idx, status, name);
            const stamp = seen[name];
            return (
              <li key={name} data-mark={mark}>
                <span className="pipe-log-mark" aria-hidden="true">
                  {mark === "done" ? <CheckIcon /> : mark === "fail" ? <FailIcon /> : mark === "now" ? (
                    <span className="pipe-spin" />
                  ) : null}
                </span>
                <div>
                  <strong>{STAGE_COPY[name].title}</strong>
                  <small>{STAGE_COPY[name].detail}</small>
                </div>
                <time>
                  {mark === "now"
                    ? "In progress"
                    : mark === "fail"
                      ? "Failed"
                      : mark === "wait"
                        ? "Pending"
                        : stamp || "Done"}
                </time>
              </li>
            );
          })}
        </ul>

        {status === "completed" && onOpenClip ? (
          <footer className="pipe-foot">
            <button type="button" onClick={onOpenClip}>
              Open Overview
            </button>
          </footer>
        ) : null}
      </section>
    </div>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
      <path
        d="M3.2 8.4 6.1 11.2 12.8 4.4"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="square"
      />
    </svg>
  );
}

function FailIcon() {
  return (
    <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
      <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function UploadGlyph() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path d="M12 16V6M8 10l4-4 4 4M5 18h14" fill="none" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}
