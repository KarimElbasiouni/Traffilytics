/**
 * Demo log-in. Credentials are published in the project README.
 */
import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { BrandMark } from "../App";
import { SourceOffer } from "../legal";

export function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  if (user) {
    return <Navigate to={destination(location.state)} replace />;
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (login(username, password)) {
      navigate(destination(location.state), { replace: true });
      return;
    }
    setError("Those credentials do not match the demo account.");
  };

  return (
    <div className="gate">
      <header className="gate-bar">
        <div className="brand">
          <BrandMark />
          <strong>Traffilytics</strong>
        </div>
      </header>
      <main className="gate-plot">
        <form className="login-form" onSubmit={onSubmit}>
          <h1>Log in</h1>
          <p className="lede">The demo account is listed in the project README.</p>
          <label>
            Username
            <input
              name="username"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(e) => {
                setUsername(e.target.value);
                setError("");
              }}
              required
            />
          </label>
          <label>
            <span className="field-head">
              Password
              <button
                type="button"
                className="login-reveal"
                onClick={() => setShowPassword((open) => !open)}
                aria-pressed={showPassword}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </span>
            <input
              name="password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                setError("");
              }}
              required
            />
          </label>
          {error ? (
            <p className="login-error" role="alert">
              {error}
            </p>
          ) : null}
          <button className="login-submit" type="submit">
            Log in
          </button>
        </form>
        <SourceOffer />
      </main>
    </div>
  );
}

function destination(state: unknown): string {
  const from = (state as { from?: { pathname?: string; search?: string } } | null)?.from;
  if (!from?.pathname || from.pathname === "/login") return "/";
  return `${from.pathname}${from.search || ""}`;
}
