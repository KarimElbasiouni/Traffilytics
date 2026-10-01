/**
 * Hand-drawn SVG figures for Overview (no chart library).
 *
 * Exports: class colors, `TimeSeriesPlot`, `DualLinePlot`, `Donut`, `Compass`,
 * `PlanView`, `Heatmap`, `MiniPath`. Pointer pick + `ChartTip` are shared
 * helpers. Color is never the only cue — captions and numeric tips sit with
 * each plot on the page.
 */
import { useId, useState, type PointerEvent } from "react";
import type { HeatCell, MapPoint, MapTrack } from "./types";

export const CLASS_COLORS: Record<string, string> = {
  bike: "#2E8B57",
  bus: "#168C8C",
  car: "#F26B38",
  other_vehicle: "#697278",
  taxi: "#E5A72A",
  truck: "#D9552B",
  unknown: "#697278",
};

export function classColor(name?: string | null) {
  return CLASS_COLORS[(name || "unknown").toLowerCase()] || CLASS_COLORS.unknown;
}

function speedColor(t: number) {
  const stops = ["#D64545", "#F26B38", "#E5A72A", "#2E8B57", "#168C8C"];
  const i = Math.max(0, Math.min(stops.length - 1, Math.floor(t * (stops.length - 1))));
  return stops[i];
}

function useItemPick<T>() {
  const [hover, setHover] = useState<T | null>(null);
  const [pinned, setPinned] = useState<T | null>(null);
  return {
    active: pinned ?? hover,
    setHover,
    togglePin: (value: T) => setPinned((prev) => (prev === value ? null : value)),
  };
}

function useChartPick() {
  return useItemPick<number>();
}

function indexFromPointer(
  event: PointerEvent<SVGSVGElement>,
  width: number,
  padL: number,
  innerW: number,
  n: number,
) {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * width;
  const t = (x - padL) / Math.max(innerW, 1);
  return Math.round(Math.max(0, Math.min(1, t)) * Math.max(n - 1, 0));
}

function fmtPlot(n: number | null | undefined, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function ChartTip({
  xPct,
  yPct,
  title,
  rows,
}: {
  xPct: number;
  yPct: number;
  title: string;
  rows: Array<{ color: string; value: string; unit: string }>;
}) {
  const flip = xPct > 72;
  return (
    <div
      className="chart-tip"
      style={{
        left: `${xPct}%`,
        top: `${yPct}%`,
        transform: flip ? "translate(-96%, -118%)" : "translate(10%, -118%)",
      }}
    >
      <strong>{title}</strong>
      {rows.map((row) => (
        <b key={`${row.color}-${row.unit}`}>
          <i style={{ background: row.color }} />
          {row.value} {row.unit}
        </b>
      ))}
    </div>
  );
}

type DualProps = {
  volume: Array<number | null | undefined>;
  speed: Array<number | null | undefined>;
  labels: string[];
  volumeLabel: string;
  speedLabel: string;
};

export function TimeSeriesPlot({
  values,
  labels,
  label,
}: {
  values: Array<number | null | undefined>;
  labels: string[];
  label: string;
}) {
  const nums = values.map((v) => (typeof v === "number" ? v : null));
  const known = nums.filter((v): v is number => v !== null);
  const fillId = useId().replace(/:/g, "");
  const strokeId = useId().replace(/:/g, "");
  const pick = useChartPick();
  if (!known.length) {
    return <p className="note">No occupancy series in this clip.</p>;
  }
  const w = 560;
  const h = 210;
  const pad = { l: 44, r: 16, t: 28, b: 28 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const n = Math.max(nums.length, 2);
  const maxV = Math.max(...known, 1);
  const xAt = (i: number) => pad.l + (i / Math.max(n - 1, 1)) * innerW;
  const yAt = (v: number) => pad.t + innerH - (v / maxV) * innerH;
  const lastI = Math.max(nums.length - 1, 0);
  const pts = nums
    .map((v, i) => (v === null ? null : `${xAt(i)},${yAt(v)}`))
    .filter(Boolean)
    .join(" ");
  const ticks = [0, Math.floor((n - 1) / 2), n - 1].filter((i, idx, arr) => arr.indexOf(i) === idx);
  const active = pick.active;
  const activeVal = active != null ? nums[active] : null;

  const onPointer = (event: PointerEvent<SVGSVGElement>) => {
    const i = indexFromPointer(event, w, pad.l, innerW, nums.length);
    pick.setHover(i);
    if (event.type === "pointerdown") pick.togglePin(i);
  };

  return (
    <figure className="chart-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        tabIndex={0}
        aria-label={label}
        onPointerMove={onPointer}
        onPointerDown={onPointer}
        onPointerLeave={() => pick.setHover(null)}
      >
        <defs>
          <linearGradient id={fillId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F26B38" stopOpacity="0.42" />
            <stop offset="100%" stopColor="#F26B38" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id={strokeId} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#F26B38" />
            <stop offset="55%" stopColor="#E5A72A" />
            <stop offset="100%" stopColor="#D9552B" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            x1={pad.l}
            x2={w - pad.r}
            y1={pad.t + innerH * (1 - t)}
            y2={pad.t + innerH * (1 - t)}
            stroke="#D9DDDF"
            strokeDasharray="3 6"
          />
        ))}
        {pts ? (
          <polygon
            fill={`url(#${fillId})`}
            points={`${pad.l},${pad.t + innerH} ${pts} ${xAt(lastI)},${pad.t + innerH}`}
          />
        ) : null}
        <polyline fill="none" stroke={`url(#${strokeId})`} strokeWidth="2.6" points={pts} />
        <text x={pad.l} y={16} fontSize={14} fontWeight={700} fontFamily="Overpass, sans-serif" fill="#697278">
          {label}
        </text>
        {ticks.map((i) => (
          <text key={i} x={xAt(i)} y={h - 6} fontSize={10} fill="#697278" textAnchor="middle">
            {labels[i] || ""}
          </text>
        ))}
        {active != null && activeVal != null ? (
          <line
            x1={xAt(active)}
            x2={xAt(active)}
            y1={pad.t}
            y2={pad.t + innerH}
            stroke="#202427"
            strokeOpacity="0.35"
            strokeWidth="1.2"
          />
        ) : null}
        {nums.map((v, i) =>
          v == null ? null : (
            <circle
              key={i}
              cx={xAt(i)}
              cy={yAt(v)}
              r={active === i ? 5 : 2.7}
              fill="#F26B38"
              stroke="#FFFFFF"
              strokeWidth={active === i ? 2 : 1.2}
            />
          ),
        )}
        <rect x={pad.l} y={pad.t} width={innerW} height={innerH} fill="transparent" />
      </svg>
      {active != null && activeVal != null ? (
        <ChartTip
          xPct={(xAt(active) / w) * 100}
          yPct={(yAt(activeVal) / h) * 100}
          title={labels[active] || "—"}
          rows={[{ color: "#F26B38", value: fmtPlot(activeVal), unit: "vehicles" }]}
        />
      ) : null}
    </figure>
  );
}

export function DualLinePlot({ volume, speed, labels, volumeLabel, speedLabel }: DualProps) {
  const vol = volume.map((v) => (typeof v === "number" ? v : null));
  const spd = speed.map((v) => (typeof v === "number" ? v : null));
  const volN = vol.filter((v): v is number => v !== null);
  const spdN = spd.filter((v): v is number => v !== null);
  const volFill = useId().replace(/:/g, "");
  const volStroke = useId().replace(/:/g, "");
  const spdFill = useId().replace(/:/g, "");
  const spdStroke = useId().replace(/:/g, "");
  const pick = useChartPick();
  if (!volN.length && !spdN.length) {
    return <p className="note">No time series in this clip.</p>;
  }
  const w = 560;
  const h = 210;
  const pad = { l: 44, r: 44, t: 28, b: 28 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const n = Math.max(volume.length, speed.length, 2);
  const maxV = Math.max(...volN, 1);
  const minS = spdN.length ? Math.min(...spdN) : 0;
  const maxS = spdN.length ? Math.max(...spdN) : 1;
  const spanS = maxS - minS || 1;
  const xAt = (i: number) => pad.l + (i / Math.max(n - 1, 1)) * innerW;
  const yVol = (v: number) => pad.t + innerH - (v / maxV) * innerH;
  const ySpd = (v: number) => pad.t + innerH - ((v - minS) / spanS) * innerH;
  const lastVol = Math.max(vol.length - 1, 0);
  const lastSpd = Math.max(spd.length - 1, 0);
  const volPts = vol
    .map((v, i) => (v === null ? null : `${xAt(i)},${yVol(v)}`))
    .filter(Boolean)
    .join(" ");
  const spdPts = spd
    .map((v, i) => (v === null ? null : `${xAt(i)},${ySpd(v)}`))
    .filter(Boolean)
    .join(" ");
  const ticks = [0, Math.floor((n - 1) / 2), n - 1].filter((i, idx, arr) => arr.indexOf(i) === idx);
  const count = Math.max(volume.length, speed.length);
  const active = pick.active;
  const onPointer = (event: PointerEvent<SVGSVGElement>) => {
    const i = indexFromPointer(event, w, pad.l, innerW, count);
    pick.setHover(i);
    if (event.type === "pointerdown") pick.togglePin(i);
  };

  return (
    <figure className="chart-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        tabIndex={0}
        aria-label={`${volumeLabel} and ${speedLabel}`}
        onPointerMove={onPointer}
        onPointerDown={onPointer}
        onPointerLeave={() => pick.setHover(null)}
      >
        <defs>
          <linearGradient id={volFill} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F26B38" stopOpacity="0.38" />
            <stop offset="100%" stopColor="#F26B38" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id={volStroke} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#F26B38" />
            <stop offset="100%" stopColor="#D9552B" />
          </linearGradient>
          <linearGradient id={spdFill} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#168C8C" stopOpacity="0.34" />
            <stop offset="100%" stopColor="#168C8C" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id={spdStroke} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#168C8C" />
            <stop offset="100%" stopColor="#2E8B57" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            x1={pad.l}
            x2={w - pad.r}
            y1={pad.t + innerH * (1 - t)}
            y2={pad.t + innerH * (1 - t)}
            stroke="#D9DDDF"
            strokeDasharray="3 6"
          />
        ))}
        {volPts ? (
          <polygon
            fill={`url(#${volFill})`}
            points={`${pad.l},${pad.t + innerH} ${volPts} ${xAt(lastVol)},${pad.t + innerH}`}
          />
        ) : null}
        {spdPts ? (
          <polygon
            fill={`url(#${spdFill})`}
            points={`${pad.l},${pad.t + innerH} ${spdPts} ${xAt(lastSpd)},${pad.t + innerH}`}
          />
        ) : null}
        <polyline fill="none" stroke={`url(#${volStroke})`} strokeWidth="2.6" points={volPts} />
        <polyline fill="none" stroke={`url(#${spdStroke})`} strokeWidth="2.4" points={spdPts} />
        <text x={pad.l} y={16} fontSize={14} fontWeight={700} fontFamily="Overpass, sans-serif" fill="#F26B38">
          {volumeLabel}
        </text>
        <text x={w - pad.r} y={16} fontSize={14} fontWeight={700} fontFamily="Overpass, sans-serif" fill="#168C8C" textAnchor="end">
          {speedLabel}
        </text>
        {ticks.map((i) => (
          <text key={i} x={xAt(i)} y={h - 6} fontSize={10} fill="#697278" textAnchor="middle">
            {labels[i] || ""}
          </text>
        ))}
        {active != null ? (
          <line
            x1={xAt(active)}
            x2={xAt(active)}
            y1={pad.t}
            y2={pad.t + innerH}
            stroke="#202427"
            strokeOpacity="0.35"
            strokeWidth="1.2"
          />
        ) : null}
        {vol.map((v, i) =>
          v == null ? null : (
            <circle
              key={`v${i}`}
              cx={xAt(i)}
              cy={yVol(v)}
              r={active === i ? 5 : 2.7}
              fill="#F26B38"
              stroke="#FFFFFF"
              strokeWidth={active === i ? 2 : 1.2}
            />
          ),
        )}
        {spd.map((v, i) =>
          v == null ? null : (
            <circle
              key={`s${i}`}
              cx={xAt(i)}
              cy={ySpd(v)}
              r={active === i ? 5 : 2.7}
              fill="#168C8C"
              stroke="#FFFFFF"
              strokeWidth={active === i ? 2 : 1.2}
            />
          ),
        )}
        <rect x={pad.l} y={pad.t} width={innerW} height={innerH} fill="transparent" />
      </svg>
      {active != null ? (
        <ChartTip
          xPct={(xAt(active) / w) * 100}
          yPct={
            (Math.min(
              vol[active] != null ? yVol(vol[active] as number) : pad.t + innerH,
              spd[active] != null ? ySpd(spd[active] as number) : pad.t + innerH,
            ) /
              h) *
            100
          }
          title={labels[active] || "—"}
          rows={[
            { color: "#F26B38", value: fmtPlot(vol[active]), unit: volumeLabel },
            { color: "#168C8C", value: fmtPlot(spd[active]), unit: speedLabel },
          ]}
        />
      ) : null}
    </figure>
  );
}

export function Donut({
  slices,
}: {
  slices: Array<{ label: string; value: number; color: string }>;
}) {
  const total = slices.reduce((s, x) => s + x.value, 0);
  const pick = useItemPick<string>();
  if (!total) return <p className="note">No class counts yet.</p>;
  const cx = 80;
  const cy = 80;
  const rOuter = 72;
  const rInner = 40;
  const gap = 0.07;
  let acc = -Math.PI / 2;
  const parts = slices.map((s) => {
    const span = (s.value / total) * Math.PI * 2;
    const g = Math.min(gap, span * 0.28);
    const a0 = acc + g / 2;
    const a1 = acc + span - g / 2;
    acc += span;
    const mid = (a0 + a1) / 2;
    const rm = (rInner + rOuter) / 2;
    return {
      ...s,
      path: ringSlice(cx, cy, rInner, rOuter, a0, a1),
      mx: cx + rm * Math.cos(mid),
      my: cy + rm * Math.sin(mid),
      name: s.label.replaceAll("_", " "),
    };
  });
  const active = parts.find((s) => s.label === pick.active) ?? null;

  return (
    <div className="donut" onPointerLeave={() => pick.setHover(null)}>
      <ul className="donut-keys">
        {parts.map((s) => (
          <li
            key={s.label}
            data-on={active?.label === s.label ? "true" : "false"}
            onPointerEnter={() => pick.setHover(s.label)}
            onPointerDown={() => pick.togglePin(s.label)}
          >
            <i style={{ background: s.color }} />
            {s.name}
          </li>
        ))}
      </ul>
      <div className="donut-plot">
        <svg viewBox="0 0 160 160" role="img" aria-label="Vehicle classes">
          {parts.map((s) => (
            <path
              key={s.label}
              className="donut-slice"
              d={s.path}
              fill={s.color}
              data-on={active?.label === s.label ? "true" : "false"}
              onPointerEnter={() => pick.setHover(s.label)}
              onPointerDown={() => pick.togglePin(s.label)}
            />
          ))}
        </svg>
        {active ? (
          <ChartTip
            xPct={(active.mx / 160) * 100}
            yPct={(active.my / 160) * 100}
            title={active.name}
            rows={[
              {
                color: active.color,
                value: `tracks: ${active.value.toLocaleString()}`,
                unit: `(${Math.round((active.value / total) * 100)}%)`,
              },
            ]}
          />
        ) : null}
      </div>
    </div>
  );
}

function ringSlice(
  cx: number,
  cy: number,
  rInner: number,
  rOuter: number,
  a0: number,
  a1: number,
) {
  if (a1 <= a0) return "";
  const x = (r: number, a: number) => cx + r * Math.cos(a);
  const y = (r: number, a: number) => cy + r * Math.sin(a);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return [
    `M ${x(rOuter, a0)} ${y(rOuter, a0)}`,
    `A ${rOuter} ${rOuter} 0 ${large} 1 ${x(rOuter, a1)} ${y(rOuter, a1)}`,
    `L ${x(rInner, a1)} ${y(rInner, a1)}`,
    `A ${rInner} ${rInner} 0 ${large} 0 ${x(rInner, a0)} ${y(rInner, a0)}`,
    "Z",
  ].join(" ");
}

const FLOW_ARMS = [
  { id: "N" as const, deg: -90, color: "#168C8C", name: "North" },
  { id: "E" as const, deg: 0, color: "#D64545", name: "East" },
  { id: "S" as const, deg: 90, color: "#E5A72A", name: "South" },
  { id: "W" as const, deg: 180, color: "#2E8B57", name: "West" },
];

export function Compass({ counts }: { counts: { N: number; E: number; S: number; W: number } }) {
  const cx = 160;
  const cy = 158;
  const max = Math.max(counts.N, counts.E, counts.S, counts.W, 1);
  const total = counts.N + counts.E + counts.S + counts.W;
  const rings = [40, 64, 88, 112];
  const softId = useId().replace(/:/g, "");
  const pick = useItemPick<(typeof FLOW_ARMS)[number]["id"]>();
  const label = `Traffic flow by heading: north ${counts.N}, east ${counts.E}, south ${counts.S}, west ${counts.W}`;
  const active = FLOW_ARMS.find((a) => a.id === pick.active) ?? null;
  const [tx, ty] = active ? polar(cx, cy, 120, active.deg) : [cx, cy];

  return (
    <div className="compass" role="img" aria-label={label} onPointerLeave={() => pick.setHover(null)}>
      <svg className="compass-svg" viewBox="0 0 320 316">
        <defs>
          <filter id={softId} x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="8" />
          </filter>
        </defs>
        {rings.map((r) => (
          <circle key={r} cx={cx} cy={cy} r={r} fill="none" stroke="#D9DDDF" strokeWidth="1.1" />
        ))}
        {FLOW_ARMS.map((arm) => {
          const n = counts[arm.id];
          const share = n / max;
          const bloomR = 16 + share * 22;
          const [bx, by] = polar(cx, cy, 78, arm.deg);
          const [dx, dy] = polar(cx, cy, 88, arm.deg);
          const [lx, ly] = polar(cx, cy, 108, arm.deg);
          const [nx, ny] = polar(cx, cy, 136, arm.deg);
          return (
            <g
              key={arm.id}
              className="compass-arm"
              data-on={active?.id === arm.id ? "true" : "false"}
              onPointerEnter={() => pick.setHover(arm.id)}
              onPointerDown={() => pick.togglePin(arm.id)}
            >
              {n > 0 ? (
                <circle
                  cx={bx}
                  cy={by}
                  r={bloomR}
                  fill={arm.color}
                  opacity={0.22 + share * 0.2}
                  filter={`url(#${softId})`}
                />
              ) : null}
              {sprayDots(n, max, arm.deg, arm.id.charCodeAt(0)).map((dot, i) => (
                <circle
                  key={`${arm.id}-${i}`}
                  cx={cx + dot.x}
                  cy={cy + dot.y}
                  r={dot.r}
                  fill={arm.color}
                  opacity={dot.o}
                />
              ))}
              <circle cx={dx} cy={dy} r={active?.id === arm.id ? 6 : 4.2} fill={arm.color} />
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={arm.color}
                fontSize="13"
                fontWeight="700"
                fontFamily="Overpass, sans-serif"
              >
                {arm.id}
              </text>
              <text
                x={nx}
                y={ny}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={arm.color}
                fontSize="18"
                fontWeight="700"
                fontFamily="Overpass, sans-serif"
              >
                {n.toLocaleString()}
              </text>
              <circle cx={dx} cy={dy} r="34" fill="transparent" />
            </g>
          );
        })}
        <circle cx={cx} cy={cy} r="22" fill="#FFFFFF" stroke="#202427" strokeWidth="1.4" />
        <path d={`M ${cx} ${cy - 11} l 3.2 6.2 h -6.4 z`} fill="#202427" />
        <text
          x={cx}
          y={cy + 9}
          textAnchor="middle"
          fill="#202427"
          fontSize="11"
          fontWeight="700"
          fontFamily="Overpass, sans-serif"
        >
          N
        </text>
      </svg>
      {active ? (
        <ChartTip
          xPct={(tx / 320) * 100}
          yPct={(ty / 316) * 100}
          title={active.name}
          rows={[
            {
              color: active.color,
              value: counts[active.id].toLocaleString(),
              unit: `tracks (${total ? Math.round((counts[active.id] / total) * 100) : 0}%)`,
            },
          ]}
        />
      ) : null}
    </div>
  );
}

function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
}

function hash32(n: number) {
  let x = n | 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

function sprayDots(count: number, max: number, deg: number, seed: number) {
  if (count <= 0) return [];
  const share = count / max;
  const n = Math.round(10 + share * 90);
  const dots: Array<{ x: number; y: number; r: number; o: number }> = [];
  let s = hash32(seed * 997 + count);
  const rnd = () => {
    s = hash32(s + 1);
    return s / 4294967295;
  };
  const rad = (deg * Math.PI) / 180;
  for (let i = 0; i < n; i += 1) {
    const along = 52 + rnd() * (36 + share * 18);
    const spread = (rnd() - 0.5) * (0.42 + share * 0.55);
    const a = rad + spread;
    dots.push({
      x: Math.cos(a) * along,
      y: Math.sin(a) * along,
      r: 0.55 + rnd() * (1.2 + share * 1.1),
      o: 0.16 + rnd() * 0.5,
    });
  }
  return dots;
}

type LayerKey = "vehicles" | "trajectories" | "speed" | "density";

function pointAtFrame(points: MapPoint[], frame: number | null | undefined) {
  if (!points.length) return undefined;
  if (frame == null) return points.at(-1);
  let best = points[0];
  for (const point of points) {
    if (point.frame <= frame) best = point;
    else break;
  }
  return best;
}

export function PlanView({
  tracks,
  heatmap,
  selectedId,
  onSelect,
  layers,
  fps,
  imageSrc,
  width,
  height,
  displayFrame,
}: {
  tracks: MapTrack[];
  heatmap: HeatCell[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  layers: Record<LayerKey, boolean>;
  fps?: number | null;
  imageSrc?: string | null;
  width?: number | null;
  height?: number | null;
  displayFrame?: number | null;
}) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const w = width && width > 0 ? width : natural?.w || 1920;
  const h = height && height > 0 ? height : natural?.h || 1080;
  if (!tracks.length && !imageSrc) {
    return <p className="note">No generated tracks to plot for this clip.</p>;
  }
  const speeds: number[] = [];
  const colored = tracks.map((track) => {
    const pts = track.points.map((p, i) => {
      let spd = p.speed ?? null;
      if (spd == null && i > 0) {
        const prev = track.points[i - 1];
        const dt = (p.frame - prev.frame) / (fps || 1);
        if (dt > 0) spd = Math.hypot(p.x - prev.x, p.y - prev.y) / dt;
      }
      if (typeof spd === "number") speeds.push(spd);
      return { ...p, spd };
    });
    return { ...track, pts };
  });
  const sMin = speeds.length ? Math.min(...speeds) : 0;
  const sMax = speeds.length ? Math.max(...speeds) : 1;
  const maxHeat = Math.max(1, ...heatmap.map((c) => c.n_points));
  const heatCols = heatmap.reduce((m, c) => Math.max(m, c.col + 1), 0) || 12;
  const heatRows = heatmap.reduce((m, c) => Math.max(m, c.row + 1), 0) || 12;
  const heatUsesPixels = heatmap.some(
    (c) => c.x0 != null && c.y0 != null && c.x1 != null && c.y1 != null,
  );

  return (
    <div className="plan-stage">
      {imageSrc ? (
        <img
          className="plan-photo"
          src={imageSrc}
          alt="Ingested intersection still"
          onLoad={(event) => {
            const img = event.currentTarget;
            if (img.naturalWidth && img.naturalHeight) {
              setNatural({ w: img.naturalWidth, h: img.naturalHeight });
            }
          }}
        />
      ) : (
        <p className="plan-missing">
          No ingested still for this clip. Tracks are drawn in pixel space without the aerial
          photograph.
        </p>
      )}
      <svg
        className="plan-draw"
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Intersection still with trajectory overlay"
      >
        {layers.density &&
          heatmap.map((cell) => {
            const t = cell.n_points / maxHeat;
            if (t <= 0) return null;
            const cw = w / heatCols;
            const ch = h / heatRows;
            const x = heatUsesPixels && cell.x0 != null ? cell.x0 : cell.col * cw;
            const y = heatUsesPixels && cell.y0 != null ? cell.y0 : cell.row * ch;
            const rw = heatUsesPixels && cell.x1 != null && cell.x0 != null ? cell.x1 - cell.x0 : cw;
            const rh = heatUsesPixels && cell.y1 != null && cell.y0 != null ? cell.y1 - cell.y0 : ch;
            return (
              <rect
                key={`${cell.row}-${cell.col}`}
                x={x}
                y={y}
                width={rw}
                height={rh}
                fill={`rgba(214, 69, 69, ${0.08 + t * 0.42})`}
              />
            );
          })}
        {layers.trajectories &&
          colored.flatMap((track) => {
            const selected = track.track_id === selectedId;
            const color = classColor(track.vehicle_type);
            if (layers.speed) {
              return track.pts.slice(1).map((p, i) => {
                const prev = track.pts[i];
                const t = ((p.spd ?? sMin) - sMin) / (sMax - sMin || 1);
                return (
                  <g key={`${track.track_id}-${i}`}>
                    <line
                      x1={prev.x}
                      y1={prev.y}
                      x2={p.x}
                      y2={p.y}
                      stroke="#1a1c1d"
                      strokeWidth={selected ? 7 : 5}
                      strokeLinecap="round"
                    />
                    <line
                      x1={prev.x}
                      y1={prev.y}
                      x2={p.x}
                      y2={p.y}
                      stroke={speedColor(t)}
                      strokeWidth={selected ? 4 : 2.4}
                      strokeLinecap="round"
                      onClick={() => onSelect(track.track_id)}
                    />
                  </g>
                );
              });
            }
            const pts = track.pts.map((p) => `${p.x},${p.y}`).join(" ");
            return [
              <polyline
                key={`halo-${track.track_id}`}
                points={pts}
                fill="none"
                stroke="#1a1c1d"
                strokeWidth={selected ? 7 : 5}
                strokeLinejoin="round"
                strokeLinecap="round"
              />,
              <polyline
                key={`line-${track.track_id}`}
                points={pts}
                fill="none"
                stroke={color}
                strokeWidth={selected ? 4 : 2.4}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={selected ? 1 : 0.92}
                onClick={() => onSelect(track.track_id)}
              />,
            ];
          })}
        {layers.vehicles &&
          colored.map((track) => {
            const last = pointAtFrame(track.pts, displayFrame);
            if (!last) return null;
            const selected = track.track_id === selectedId;
            return (
              <g
                key={`v${track.track_id}`}
                onClick={() => onSelect(track.track_id)}
                style={{ cursor: "pointer" }}
              >
                <circle
                  cx={last.x}
                  cy={last.y}
                  r={selected ? 9 : 6}
                  fill={classColor(track.vehicle_type)}
                  stroke="#f4f1ea"
                  strokeWidth={selected ? 2.2 : 1.6}
                />
              </g>
            );
          })}
      </svg>
    </div>
  );
}

export function Heatmap({ cells }: { cells: HeatCell[] }) {
  if (!cells.length) {
    return <p className="note">No occupancy grid for this clip.</p>;
  }
  const cols = cells.reduce((m, c) => Math.max(m, c.col + 1), 0) || 12;
  const maxN = Math.max(1, ...cells.map((c) => c.n_points));
  return (
    <figure className="chart-wrap">
      <div
        className="heat"
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
        role="img"
        aria-label="Occupancy heatmap of trajectory points"
      >
        {cells.map((c) => {
          const t = c.n_points / maxN;
          const hue = 120 - Math.round(t * 120);
          return (
            <button
              key={`${c.row}-${c.col}`}
              type="button"
              title={`${c.n_points} points, mean speed ${c.mean_speed ?? "n/a"}`}
              style={{ background: `hsl(${hue} 62% ${92 - t * 42}%)` }}
            />
          );
        })}
      </div>
      <figcaption className="chart-caption">
        Green is sparse, red is dense. Max {maxN} points in a cell.
      </figcaption>
    </figure>
  );
}

export function MiniPath({ track }: { track: MapTrack | undefined }) {
  if (!track?.points.length) return <p className="note">Select a track on the plan view.</p>;
  const w = 280;
  const h = 120;
  const xs = track.points.map((p) => p.x);
  const ys = track.points.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (x: number) => 8 + ((x - x0) / (x1 - x0 || 1)) * (w - 16);
  const sy = (y: number) => 8 + ((y - y0) / (y1 - y0 || 1)) * (h - 16);
  const d = track.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
  const last = track.points.at(-1)!;
  return (
    <svg className="mini-path" viewBox={`0 0 ${w} ${h}`} aria-label={`Track ${track.track_id}`}>
      <path d={d} fill="none" stroke={classColor(track.vehicle_type)} strokeWidth="2.4" />
      <circle cx={sx(last.x)} cy={sy(last.y)} r="5" fill="#2E8B57" />
    </svg>
  );
}
