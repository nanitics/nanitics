import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backendUrl = process.env.OBSERVATORY_BACKEND ?? "http://localhost:8001";

export default defineConfig({
  root: "dev",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api/observatory": backendUrl,
    },
  },
  test: {
    root: ".",
    include: ["tests/**/*.{test,spec}.{ts,tsx}"],
    environment: "jsdom",
    setupFiles: ["tests/setup.ts"],
  },
});
