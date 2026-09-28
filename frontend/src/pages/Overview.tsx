import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import {
  classColor,
  Compass,
  Donut,
  DualLinePlot,
  MiniPath,
  TimeSeriesPlot,
} from "../charts";
import { NeedClip, useClip } from "../clip";
import type { EventRow, FlowWindow, MapTrack } from "../types";

function fmt(n: number | null | undefined, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function OverviewPage() {
  return <NeedClip>{(clip) => <OverviewBody clip={clip} />}</NeedClip>;
}

function OverviewBody({ clip }: { clip: string }) {
  const { jobId } = useClip();
  const overview = useQuery({ queryKey: ["overview", clip], queryFn: () => api.overview(clip) });
  const flow = useQuery({ queryKey: ["flow", clip], queryFn: () => api.flow(clip) });
  const events = useQuery({ queryKey: ["events", clip], queryFn: () => api.events(clip) });
  const vehicles = useQuery({ queryKey: ["vehicles", clip], queryFn: () => api.vehicles(clip) });
  const map = useQuery({ queryKey: ["map", clip], queryFn: () => api.map(clip) });
  const [overlayFailed, setOverlayFailed] = useState(false);
  const btn = useQuery({ queryKey: ["btn", clip], queryFn: () => api.bottlenecks(clip) });
  const imb = useQuery({ queryKey: ["imb", clip], queryFn: () => api.imbalance(clip) });
  const videos = useQuery({ queryKey: ["videos"], queryFn: api.videos });
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "queued" || s === "processing" ? 1200 : false;
    },
  });

  useEffect(() => {
    setOverlayFailed(false);
  }, [clip]);

  const tracks = map.data?.tracks || [];
  const activeId = tracks[0]?.track_id ?? null;
  const selected = tracks.find((t) => t.track_id === activeId);
  const vehicle = (vehicles.data?.vehicles || []).find((v) => v.track_id === activeId);

  const compass = useMemo(() => headingCounts(tracks), [tracks]);
  const slices = useMemo(() => {
    const counts = new Map<string, number>();
    for (const row of vehicles.data?.vehicles || []) {
      const key = row.vehicle_type || "unknown";
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([label, value]) => ({ label, value, color: classColor(label) }));
  }, [vehicles.data]);

  if (overview.isPending) return <p className="note">Loading console…</p>;
  if (overview.isError) return <p className="err">{(overview.error as Error).message}</p>;

  const d = overview.data;
  const listed = (videos.data?.videos || []).find((v) => v.video_id === clip);
  const units = d.units || {};
  const windows = flow.data?.flow.windows || [];
  const eventRows = events.data?.events || [];
  const state = (d.traffic_state || "unknown").toLowerCase();
  const findings = clipFindings({
    state,
    windows,
    events: eventRows,
    headings: compass,
    bottleneck: btn.data?.bottleneck,
    imbalance: imb.data?.imbalance,
  });
  const pixelBased = units.labelled_as_physical !== true;

  return (
    <div className="console">
      <section className="kpis" aria-label="Scene totals">
        <Kpi tone="blue" label="Total tracks" value={(d.n_vehicles ?? 0).toLocaleString()} hint="Generated IDs" />
        <Kpi
          tone="cyan"
          label="Average speed"
          value={fmt(d.mean_speed)}
          unit={units.speed || "px/s"}
          hint={pixelBased ? "pixel-based" : "ground scale"}
        />
        <Kpi tone={stateTone(state)} label="Congestion level" value={state} hint="Majority window state" />
        <Kpi tone="rose" label="Active events" value={String(d.n_events ?? eventRows.length)} hint="Rule-based" />
        <Kpi
          tone="violet"
          label="Volume"
          value={fmt(d.vehicles_per_minute)}
          unit="veh/min"
        />
      </section>

      <section className="hero">
        <article className="card plan-card">
          <header>
            <h2>Tracked clip</h2>
            <ClipMeta
              clip={clip}
              duration={d.duration ?? listed?.duration}
              status={job.data?.video_id === clip ? job.data.status : d.status || listed?.status}
              stage={job.data?.video_id === clip ? job.data.stage : undefined}
            />
          </header>
          <video
            key={clip}
            className="overlay-video"
            controls
            playsInline
            src={api.overlayUrl(clip)}
            onError={() => setOverlayFailed(true)}
            aria-label="Clip with oriented boxes and track IDs"
          />
          <p className="chart-caption">
            Each box is a detection on that frame. A track id is shown when the
            tracker kept the vehicle; untracked detections are still boxed.
          </p>
          {overlayFailed ? (
            <p className="note">
              Overlay video is still building or this clip has no ingested frames. Re-process
              the clip if it stays blank.
            </p>
          ) : null}
          <p className="legend">
            {slices.slice(0, 6).map((s) => (
              <span key={s.label}>
                <i style={{ background: s.color }} />
                {s.label.replaceAll("_", " ")}
              </span>
            ))}
          </p>
        </article>

        <div className="hero-side">
          <article className="card">
            <header>
              <h2>Traffic flow</h2>
              <span className="flow-north" title="Up on this figure is frame north">
                N
              </span>
            </header>
            <Compass counts={compass} />
            <p className="chart-caption">
              First-to-last heading of each track in the frame.
            </p>
          </article>
          <article className="card mix-card">
            <header>
              <h2>Tracked vehicles by class</h2>
            </header>
            <Donut slices={slices} />
            <p className="chart-total">
              {slices.reduce((sum, s) => sum + s.value, 0).toLocaleString()}
              <span> tracks</span>
            </p>
            <p className="chart-caption">
              Slice size is the share of generated track IDs in that detector class.
            </p>
          </article>
        </div>

        <article className="card events-card">
          <header>
            <h2>Recent events</h2>
            <span>{eventRows.length}</span>
          </header>
          {!eventRows.length ? (
            <p className="note">No events stored for this clip.</p>
          ) : (
            <ul className="event-list">
              {eventRows.slice(0, 8).map((e, i) => (
                <li key={e.event_id || i} data-sev={(e.severity || "low").toLowerCase()}>
                  <strong>{pretty(e.event_type || e.type)}</strong>
                  <span>
                    {e.timestamp || e.time || "—"}
                    {e.location || e.zone_id ? ` · ${e.location || e.zone_id}` : ""}
                    {e.track_id != null ? ` · #${e.track_id}` : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </article>
      </section>

      <section className="bottom">
        <article className="card">
          <header>
            <h2>Traffic volume &amp; speed</h2>
            {pixelBased ? <span className="unit-tag">pixel-based</span> : null}
          </header>
          <DualLinePlot
            volume={windows.map((w) => w.volume_per_min)}
            speed={windows.map((w) => w.mean_speed)}
            labels={windows.map((w) => `${w.t0.toFixed(0)}s`)}
            volumeLabel="veh/min"
            speedLabel={units.speed || "px/s"}
          />
          <p className="chart-caption">
            Pixel speed includes camera motion on a moving drone. Use occupancy (right) for how
            busy the frame is.
          </p>
        </article>
        <article className="card">
          <header>
            <h2>Vehicles in view</h2>
          </header>
          <TimeSeriesPlot
            values={windows.map((w) => w.occupancy_mean)}
            labels={windows.map((w) => `${w.t0.toFixed(0)}s`)}
            label="mean vehicles visible"
          />
          <p className="chart-caption">
            Count of vehicles on screen over time. Unlike a spatial heatmap, this does not smear
            when the camera pans or the drone flies.
          </p>
        </article>
        <article className="card">
          <header>
            <h2>Vehicle trajectories</h2>
            <span>{activeId != null ? `#${activeId}` : "—"}</span>
          </header>
          <MiniPath track={selected} />
          <p className="chart-caption">
            Pixel path of one track. On a moving drone this mixes vehicle motion with camera
            motion — use it to inspect identity, not ground position.
          </p>
          <dl className="facts">
            <div>
              <dt>Type</dt>
              <dd>{vehicle?.vehicle_type || selected?.vehicle_type || "—"}</dd>
            </div>
            <div>
              <dt>Points</dt>
              <dd>{selected?.n_points ?? selected?.points.length ?? "—"}</dd>
            </div>
            <div>
              <dt>Entry</dt>
              <dd>{vehicle?.entry_time || "—"}</dd>
            </div>
            <div>
              <dt>Exit</dt>
              <dd>{vehicle?.exit_time || "—"}</dd>
            </div>
          </dl>
          {imb.data?.imbalance.configured ? (
            <div className="shares">
              {(imb.data.imbalance.lanes || []).map((row) => (
                <div className="share" key={row.lane}>
                  <span>{row.lane}</span>
                  <div className="bar" aria-hidden="true">
                    <span style={{ width: `${Math.round(row.share * 100)}%` }} />
                  </div>
                  <span>{Math.round(row.share * 100)}%</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="note">{imb.data?.imbalance.note || "Lane polygons not configured."}</p>
          )}
        </article>
        <article className="card">
          <header>
            <h2>Insights</h2>
          </header>
          {!findings.length ? (
            <p className="note">Nothing stands out beyond the charts on this page.</p>
          ) : (
            <ul className="insight-list">
              {findings.map((item) => (
                <li key={item.title}>
                  <strong>{item.title}</strong>
                  <span>{item.detail}</span>
                </li>
              ))}
            </ul>
          )}
        </article>
      </section>
    </div>
  );
}

function ClipMeta({
  clip,
  duration,
  status,
  stage,
}: {
  clip: string;
  duration?: number | null;
  status?: string;
  stage?: string;
}) {
  const mark = (status || "idle").toLowerCase();
  return (
    <div className="meta-chips">
      <span>{clip}</span>
      <span>{duration != null ? `${duration.toFixed(1)} s` : "No duration"}</span>
      <span data-status={mark} className="status-pill">
        {labelStatus(mark, stage)}
      </span>
    </div>
  );
}

function labelStatus(status: string, stage?: string) {
  if (status === "completed") return "Processing complete";
  if (status === "processing") return stage ? `Processing: ${stage}` : "Processing";
  if (status === "queued") return "Queued";
  if (status === "failed") return "Failed";
  return "Idle";
}

function Kpi({
  label,
  value,
  hint,
  tone,
  unit,
}: {
  label: string;
  value: string;
  hint?: string;
  tone: string;
  unit?: string;
}) {
  return (
    <article className="kpi" data-tone={tone}>
      <span>{label}</span>
      <b>
        {value}
        {unit ? <span className="unit-inline">{unit}</span> : null}
      </b>
      {hint ? <small>{hint}</small> : null}
    </article>
  );
}

function stateTone(state: string) {
  if (state === "free") return "emerald";
  if (state === "congested") return "amber";
  if (state === "queued") return "rose";
  return "slate";
}

function pretty(value?: string | null) {
  return (value || "—").replaceAll("_", " ");
}

type Finding = { title: string; detail: string };

function clipFindings(input: {
  state: string;
  windows: FlowWindow[];
  events: EventRow[];
  headings: { N: number; E: number; S: number; W: number };
  bottleneck?: {
    configured?: boolean;
    primary?: { zone?: string; cause?: string } | null;
  };
  imbalance?: {
    configured?: boolean;
    lanes?: Array<{ lane: string; share: number }>;
    imbalance_index?: number | null;
  };
}): Finding[] {
  const out: Finding[] = [];
  const { windows, events, headings } = input;

  const byType = new Map<string, number>();
  for (const row of events) {
    const key = pretty(row.event_type || row.type);
    if (key === "—") continue;
    byType.set(key, (byType.get(key) || 0) + 1);
  }
  if (events.length) {
    const parts = [...byType.entries()].map(([name, n]) => `${n} ${name}`);
    out.push({
      title: events.length === 1 ? "1 event stored" : `${events.length} events stored`,
      detail: parts.join(", ") + ".",
    });
  }

  const primary = input.bottleneck?.configured ? input.bottleneck.primary : null;
  if (primary?.zone) {
    out.push({
      title: `Slowdown in ${primary.zone}`,
      detail: primary.cause
        ? `The bottleneck rule named ${pretty(primary.cause)}.`
        : "A configured zone scored highest for delay.",
    });
  }

  if (windows.length) {
    const queued = windows.filter((w) => w.state === "queued").length;
    const congested = windows.filter((w) => w.state === "congested").length;
    const free = windows.filter((w) => w.state === "free").length;
    if (queued / windows.length >= 0.4) {
      out.push({
        title: "Queueing for much of the clip",
        detail: `${queued} of ${windows.length} time windows were queued.`,
      });
    } else if (congested / windows.length >= 0.4) {
      out.push({
        title: "Congested windows dominate",
        detail: `${congested} of ${windows.length} time windows were congested.`,
      });
    } else if (free / windows.length >= 0.5) {
      out.push({
        title: "Mostly free flow",
        detail: `${free} of ${windows.length} time windows were free.`,
      });
    } else if (input.state && input.state !== "unknown") {
      out.push({
        title: `Overall state is ${input.state}`,
        detail: `Free ${free}, congested ${congested}, queued ${queued} of ${windows.length} windows.`,
      });
    }
  }

  const headingTotal = headings.N + headings.E + headings.S + headings.W;
  if (headingTotal > 0) {
    const ranked: Array<[string, number]> = [
      ["north", headings.N],
      ["east", headings.E],
      ["south", headings.S],
      ["west", headings.W],
    ];
    ranked.sort((a, b) => b[1] - a[1]);
    const [dir, n] = ranked[0];
    const pct = Math.round((n / headingTotal) * 100);
    if (pct >= 40) {
      out.push({
        title: `${dir[0].toUpperCase()}${dir.slice(1)} is the main heading`,
        detail: `${n} of ${headingTotal} tracks (${pct}%) moved ${dir} in the frame.`,
      });
    }
  }

  const occupied = windows.filter((w) => typeof w.occupancy_mean === "number");
  if (occupied.length) {
    const peak = occupied.reduce((best, w) =>
      w.occupancy_mean > best.occupancy_mean ? w : best,
    );
    if (peak.occupancy_mean > 0) {
      out.push({
        title: `Busiest around ${peak.t0.toFixed(0)}s`,
        detail: `About ${peak.occupancy_mean.toFixed(1)} vehicles were on screen in that window.`,
      });
    }
  }

  const lanes = input.imbalance?.configured ? input.imbalance.lanes || [] : [];
  if (lanes.length >= 2 && (input.imbalance?.imbalance_index ?? 0) >= 0.15) {
    const ranked = [...lanes].sort((a, b) => b.share - a.share);
    out.push({
      title: `${ranked[0].lane} carries more traffic`,
      detail: `${Math.round(ranked[0].share * 100)}% vs ${Math.round(ranked[1].share * 100)}% on ${ranked[1].lane}.`,
    });
  }

  return out.slice(0, 4);
}

function headingCounts(tracks: MapTrack[]) {
  const counts = { N: 0, E: 0, S: 0, W: 0 };
  for (const track of tracks) {
    const a = track.points[0];
    const b = track.points.at(-1);
    if (!a || !b) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    if (dx === 0 && dy === 0) continue;
    if (Math.abs(dx) >= Math.abs(dy)) counts[dx >= 0 ? "E" : "W"] += 1;
    else counts[dy >= 0 ? "S" : "N"] += 1;
  }
  return counts;
}
