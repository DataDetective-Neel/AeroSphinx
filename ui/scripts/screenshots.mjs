import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const OUT = path.resolve("../docs/screenshots");
const PAGES = [
  ["/", "command-center"],
  ["/research-lab", "research-lab"],
  ["/data-observatory", "data-observatory"],
  ["/validation-matrix", "validation-matrix"],
  ["/defect-register", "defect-register"],
  ["/model-intelligence", "model-intelligence"],
  ["/forecast-studio", "forecast-studio"],
  ["/atmospheric-explorer", "atmospheric-explorer"],
  ["/reports", "reports"],
  ["/settings", "settings"],
];

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium" });

for (const theme of ["dark", "light"]) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  await ctx.addInitScript((t) => localStorage.setItem("aerosphinx.theme", t), theme);
  const page = await ctx.newPage();
  for (const [route, name] of PAGES) {
    await page.goto(`http://127.0.0.1:5173${route}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(OUT, `${name}-${theme}.png`) });
  }
  await ctx.close();
}

// One mobile capture to demonstrate the responsive shell.
const mobile = await browser.newContext({
  viewport: { width: 390, height: 844 },
  isMobile: true,
  deviceScaleFactor: 2,
});
await mobile.addInitScript(() => localStorage.setItem("aerosphinx.theme", "dark"));
const mp = await mobile.newPage();
await mp.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
await mp.getByRole("button", { name: /toggle navigation/i }).click();
await mp.waitForTimeout(300);
await mp.screenshot({ path: path.join(OUT, "mobile-navigation-dark.png") });
await mobile.close();

await browser.close();
console.log("screenshots written to", OUT);