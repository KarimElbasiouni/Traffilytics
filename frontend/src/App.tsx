/**
 * Shell: left chainage (Overview / Upload), title bar, clip picker, outlet.
 *
 * Nav links keep `?clip=` so switching stations does not drop the selection.
 * The demo log-in gate lives in the router; this shell does not call the API
 * with credentials.
 */
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { useAuth } from "./auth";
import { useClip } from "./clip";

const NAV = [
  { to: "/", label: "Overview", icon: IconGrid, end: true },
  { to: "/upload", label: "Upload", icon: IconUpload, end: false },
];

export function App() {
  const { clip, setClip } = useClip();
  const { logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const videos = useQuery({ queryKey: ["videos"], queryFn: api.videos });
  const page =
    NAV.find((item) => (item.end ? location.pathname === item.to : location.pathname.startsWith(item.to)))
      ?.label || "Overview";

  return (
    <div className="shell">
      <aside className="rail" aria-label="Primary">
        <div className="brand">
          <BrandMark />
          <strong>Traffilytics</strong>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={clip ? `${item.to}?clip=${encodeURIComponent(clip)}` : item.to}
              end={item.end}
            >
              <item.icon />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <p className="rail-foot">UAV-OBB · AGPL-3.0</p>
      </aside>
      <div className="stage">
        <header className="topbar">
          <h1 className="page-title">{page}</h1>
          <div className="topbar-actions">
            <label className="clip-pick">
              Site
              <select
                value={clip}
                onChange={(e) => setClip(e.target.value)}
                aria-label="Select a processed clip"
              >
                <option value="">None selected</option>
                {(videos.data?.videos || []).map((v) => (
                  <option key={v.video_id} value={v.video_id}>
                    {v.site || v.video_id}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="quiet sign-out"
              onClick={() => {
                logout();
                navigate("/login", { replace: true });
              }}
            >
              Log out
            </button>
          </div>
        </header>
        <main className="plot">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export function BrandMark() {
  const cx = 16;
  const cy = 16;
  const rOuter = 15;
  const rInner = 8.1;
  const gap = 0.07;
  const slices = [
    { color: "#F26B38", span: 1.72 },
    { color: "#168C8C", span: 2.48 },
    { color: "#E5A72A", span: 2.083 },
  ];
  let acc = -Math.PI / 2;
  const wedges = slices.map((s) => {
    const a0 = acc + gap / 2;
    const a1 = acc + s.span - gap / 2;
    acc += s.span;
    return { ...s, d: ringWedge(cx, cy, rInner, rOuter, a0, a1) };
  });

  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      {wedges.map((s) => (
        <path key={s.color} d={s.d} fill={s.color} />
      ))}
      <rect x="11.05" y="16.9" width="2.4" height="5.5" rx="1.15" fill="#F26B38" />
      <rect x="14.8" y="13.7" width="2.4" height="8.7" rx="1.15" fill="#168C8C" />
      <rect x="18.55" y="11.1" width="2.4" height="11.3" rx="1.15" fill="#E5A72A" />
    </svg>
  );
}

function ringWedge(
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

function IconGrid() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="3" width="8" height="8" rx="1.5" />
      <rect x="13" y="3" width="8" height="8" rx="1.5" />
      <rect x="3" y="13" width="8" height="8" rx="1.5" />
      <rect x="13" y="13" width="8" height="8" rx="1.5" />
    </svg>
  );
}

function IconUpload() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V6" />
      <path d="M8 10l4-4 4 4" />
      <path d="M5 18h14" />
    </svg>
  );
}
