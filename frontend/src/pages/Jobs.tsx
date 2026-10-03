/**
 * Upload station: drop a video, enqueue processing, pick an existing clip.
 *
 * The POST returns immediately with `video_id` / `job_id`. Detection runs in
 * a worker; this page only polls `GET /jobs/:id` and opens `PipelineProgress`.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useClip } from "../clip";
import { PipelineProgress } from "../PipelineProgress";

const ACCEPT = [".mp4", ".avi", ".mov", ".mkv"];

function fileOk(file: File) {
  const name = file.name.toLowerCase();
  return ACCEPT.some((ext) => name.endsWith(ext));
}

function fmtSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function JobsPage() {
  const { clip, setClip, jobId, setJobId } = useClip();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [site, setSite] = useState("");
  const [msg, setMsg] = useState("");
  const [watch, setWatch] = useState(false);
  const [hold, setHold] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);

  const videos = useQuery({ queryKey: ["videos"], queryFn: api.videos });
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "queued" || s === "processing" ? 900 : false;
    },
  });

  const takeFile = (next: File | null) => {
    if (next && !fileOk(next)) {
      setMsg("Use an MP4, MOV, AVI, or MKV file.");
      return;
    }
    setMsg("");
    setFile(next);
    if (fileRef.current && !next) fileRef.current.value = "";
  };

  const onDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setOver(false);
    takeFile(event.dataTransfer.files[0] ?? null);
  };

  const upload = useMutation({
    mutationFn: async () => {
      const chosen = file;
      if (!chosen) throw new Error("Choose a video file first.");
      const body = new FormData();
      body.append("file", chosen);
      if (site.trim()) body.append("site", site.trim());
      return api.upload(body);
    },
    onMutate: () => {
      setWatch(true);
      setHold(true);
      setMsg("");
    },
    onSuccess: (data) => {
      setClip(data.video_id);
      setJobId(data.job_id);
      setHold(false);
      void qc.invalidateQueries({ queryKey: ["videos"] });
    },
    onError: (err: Error) => {
      setHold(false);
      setWatch(false);
      setMsg(err.message);
    },
  });

  const reprocess = useMutation({
    mutationFn: () => api.process(clip),
    onMutate: () => {
      setWatch(true);
      setHold(true);
    },
    onSuccess: (data) => {
      setJobId(data.job_id);
      setHold(false);
    },
    onError: (err: Error) => {
      setHold(false);
      setWatch(false);
      setMsg(err.message);
    },
  });

  return (
    <div className="upload-page">
      {watch ? (
        <PipelineProgress
          job={hold ? undefined : job.data}
          uploading={upload.isPending || hold}
          onClose={() => setWatch(false)}
          onOpenClip={() => {
            setWatch(false);
            navigate(clip ? `/?clip=${encodeURIComponent(clip)}` : "/");
          }}
        />
      ) : null}

      <article className="intake">
        <header className="intake-head">
          <span className="intake-badge" aria-hidden="true">
            <CloudIcon />
          </span>
          <div>
            <h2>Upload a clip</h2>
            <p>
              Add overhead footage of a traffic intersection, or pick a clip that is already on this
              site from the list below. Analysis starts after you upload.
            </p>
          </div>
        </header>

        <form
          className="intake-form"
          onSubmit={(e) => {
            e.preventDefault();
            upload.mutate();
          }}
        >
          <input
            ref={fileRef}
            className="sr-only"
            type="file"
            accept={ACCEPT.join(",")}
            onChange={(e) => takeFile(e.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            className="dropzone"
            data-over={over ? "true" : "false"}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={onDrop}
          >
            <CloudIcon />
            <strong>Drag and drop your video file here</strong>
            <span>or click to browse</span>
            <small>MP4, MOV, AVI, MKV</small>
          </button>

          {file ? (
            <div className="file-chip">
              <span className="file-glyph" aria-hidden="true">
                <FileIcon />
              </span>
              <div>
                <b>{file.name}</b>
                <small>{fmtSize(file.size)}</small>
              </div>
              <button type="button" className="pipe-x" onClick={() => takeFile(null)} aria-label="Remove file">
                ×
              </button>
            </div>
          ) : null}

          <label>
            Site label (optional)
            <input
              value={site}
              onChange={(e) => setSite(e.target.value)}
              placeholder="arterial_north"
            />
          </label>

          <button type="submit" className="intake-go" disabled={!file || upload.isPending}>
            <UploadIcon />
            {upload.isPending ? "Uploading" : "Upload and analyze"}
          </button>
          {msg ? <p className="err">{msg}</p> : null}
        </form>
      </article>

      <p className="path-or" role="separator">
        or use an existing clip
      </p>

      <article className="card clips-card">
        <header>
          <div>
            <h2>Existing clips</h2>
            <p>Click a name to select it. Overview uses the selected clip.</p>
          </div>
          <button type="button" className="quiet" disabled={!clip || reprocess.isPending} onClick={() => reprocess.mutate()}>
            Re-process selected
          </button>
        </header>
        {videos.isError ? (
          <p className="err">{(videos.error as Error).message}</p>
        ) : !videos.data?.videos.length ? (
          <p className="note">None on this site yet. Upload a video above.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>clip</th>
                <th>status</th>
                <th>site</th>
                <th>size</th>
                <th>duration</th>
              </tr>
            </thead>
            <tbody>
              {videos.data.videos.map((v) => (
                <tr key={v.video_id} data-on={v.video_id === clip ? "true" : "false"}>
                  <td>
                    <button type="button" className="quiet" onClick={() => setClip(v.video_id)}>
                      {v.video_id}
                    </button>
                  </td>
                  <td>{v.status}</td>
                  <td>{v.site || "—"}</td>
                  <td>{v.resolution || "—"}</td>
                  <td>{v.duration != null ? `${v.duration.toFixed(1)} s` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </article>
    </div>
  );
}

function CloudIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <path
        d="M7 17h11a4 4 0 0 0 .4-8 6 6 0 0 0-11.4-1.5A4 4 0 0 0 7 17Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <path d="M12 14V8M9.5 10.5 12 8l2.5 2.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path d="M7 3h7l5 5v13H7z" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="M14 3v5h5" fill="none" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path d="M12 16V6M8 10l4-4 4 4M5 18h14" fill="none" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}
