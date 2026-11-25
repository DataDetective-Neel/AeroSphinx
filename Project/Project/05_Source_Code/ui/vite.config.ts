/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8787",
        changeOrigin: true,
      },
    },
    // @ts-ignore: historyApiFallback not in ServerOptions type for this Vite version
    historyApiFallback: true,
  },
  build: {
    rollupOptions: {
      output: {
        // Three stable chunks: the React runtime, the charting engine and
        // everything else. ECharts is ~600 kB of the old single bundle and is
        // only needed on pages that draw, so it must not block first paint.
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("echarts") || id.includes("zrender")) return "echarts";
          if (id.includes("@tanstack")) return "query";
          if (/node_modules\/(react|react-dom|react-router|scheduler)/.test(id)) {
            return "react";
          }
          return undefined;
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    css: false,
  },
});