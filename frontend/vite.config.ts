import { execSync } from "node:child_process";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const OBB_V1 = "1100b33da026c692a5595b3628cf5d5c75a3366e";

function sourceCommit(): string {
  try {
    return execSync("git rev-parse HEAD", { encoding: "utf8" }).trim() || OBB_V1;
  } catch {
    return OBB_V1;
  }
}

export default defineConfig({
  define: {
    __SOURCE_COMMIT__: JSON.stringify(sourceCommit()),
  },
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
