import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build the Observatory SPA into the Python package so the wheel ships it.
// `create_observatory_ui_router` defaults `static_dir` to this directory
// via `importlib.resources`, so a fresh `pip install nanitics` + `mount_observatory(...)`
// renders the UI with no frontend toolchain on the consumer side.
export default defineConfig({
  root: "dev",
  base: "./",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../../nanitics/observatory/ui_assets",
    emptyOutDir: true,
  },
});
