/**
 * Demo sign-in for the dashboard pages.
 *
 * The check lives in this browser tab. It does not authorize API calls
 * (NFR-SEC-002: the local API stays open).
 */
import { createContext, useContext, useState, type ReactNode } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";

export const DEMO_USER = "admin101";
export const DEMO_PASS = "admin123";

const SESSION_KEY = "traffilytics.demoUser";

type AuthState = {
  user: string;
  login: (username: string, password: string) => boolean;
  logout: () => void;
};

const AuthCtx = createContext<AuthState | null>(null);

function readUser(): string {
  const saved = sessionStorage.getItem(SESSION_KEY);
  return saved === DEMO_USER ? saved : "";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState(readUser);

  const login = (username: string, password: string) => {
    const ok = username.trim() === DEMO_USER && password.trim() === DEMO_PASS;
    if (!ok) return false;
    sessionStorage.setItem(SESSION_KEY, DEMO_USER);
    setUser(DEMO_USER);
    return true;
  };

  const logout = () => {
    sessionStorage.removeItem(SESSION_KEY);
    setUser("");
  };

  return <AuthCtx.Provider value={{ user, login, logout }}>{children}</AuthCtx.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error("useAuth requires AuthProvider");
  return ctx;
}

export function RequireAuth() {
  const { user } = useAuth();
  const location = useLocation();
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return <Outlet />;
}
