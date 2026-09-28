import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { App } from "./App";
import { ClipProvider } from "./clip";
import { JobsPage } from "./pages/Jobs";
import { OverviewPage } from "./pages/Overview";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <ClipProvider>
          <Routes>
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
          </Routes>
        </ClipProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
