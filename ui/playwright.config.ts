import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end configuration. Uses the locally installed Chrome rather than
 * downloading a separate browser binary.
 *
 * Requires the artifact API on :8787 and the Vite dev server on :5173.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    screenshot: "only-on-failure",
    trace: "off",
    launchOptions: { executablePath: "/usr/bin/chromium" },
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});