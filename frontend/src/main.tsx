/**
 * SPA entry: React Query, router, clip context, and station routes.
 *
 * `/login` is the demo gate. Real pages are `/` (Overview) and `/upload`.
 * Older analytics paths redirect to Overview so bookmarks still land on the console.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { App } from "./App";
import { AuthProvider, RequireAuth } from "./auth";
import { ClipProvider } from "./clip";
import { JobsPage } from "./pages/Jobs";
import { LoginPage } from "./pages/Login";
import { OverviewPage } from "./pages/Overview";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <AuthProvider>
          <ClipProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route element={<RequireAuth />}>
                <Route element={<App />}>
                  <Route path="/" element={<OverviewPage />} />
                  <Route path="/upload" element={<JobsPage />} />
                  <Route path="/overview" element={<Navigate to="/" replace />} />
                  <Route path="/flow" element={<Navigate to="/" replace />} />
                  <Route path="/bottleneck" element={<Navigate to="/" replace />} />
                  <Route path="/imbalance" element={<Navigate to="/" replace />} />
                  <Route path="/events" element={<Navigate to="/" replace />} />
                  <Route path="/report" element={<Navigate to="/" replace />} />
                </Route>
              </Route>
            </Routes>
          </ClipProvider>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
