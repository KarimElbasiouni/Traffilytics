/**
 * Selected clip and latest job id for the dashboard session.
 *
 * Clip id lives in `?clip=` (and localStorage). Job id is local only.
 * `NeedClip` is the empty-state gate for Overview: no metrics until a clip
 * is chosen.
 */
import { createContext, useContext, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";

type ClipState = {
  clip: string;
  setClip: (id: string) => void;
  jobId: string;
  setJobId: (id: string) => void;
};

const ClipCtx = createContext<ClipState | null>(null);

const CLIP_KEY = "traffilytics.videoId";
const JOB_KEY = "traffilytics.jobId";

export function ClipProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams();
  const stored = localStorage.getItem(CLIP_KEY) || "";
  const clip = params.get("clip") || stored;
  const [jobId, setJobIdState] = useState(() => localStorage.getItem(JOB_KEY) || "");

  const setClip = (id: string) => {
    localStorage.setItem(CLIP_KEY, id);
    const next = new URLSearchParams(params);
    if (id) next.set("clip", id);
    else next.delete("clip");
    setParams(next, { replace: true });
  };
  const setJobId = (id: string) => {
    localStorage.setItem(JOB_KEY, id);
    setJobIdState(id);
  };

  return (
    <ClipCtx.Provider value={{ clip, setClip, jobId, setJobId }}>{children}</ClipCtx.Provider>
  );
}

export function useClip(): ClipState {
  const ctx = useContext(ClipCtx);
  if (!ctx) throw new Error("useClip requires ClipProvider");
  return ctx;
}

export function NeedClip({ children }: { children: (clip: string) => ReactNode }) {
  const { clip } = useClip();
  if (!clip) {
    return (
      <div className="empty-console">
        <h2>Pick a clip to open the console</h2>
        <p>
          Overview shows the boxed clip, occupancy over time, events, and trajectories. Upload a
          video, or choose one from the site menu.
        </p>
        <Link className="btn" to="/upload">
          Go to Upload
        </Link>
      </div>
    );
  }
  return <>{children(clip)}</>;
}
