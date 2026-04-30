import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  root: "dev",
  base: "./",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../dist-embed",
    emptyOutDir: true,
  },
});
