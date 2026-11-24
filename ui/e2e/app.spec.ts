import { test, expect } from "@playwright/test";

const MODULES = [
  ["/", "Command Center"],
  ["/research-lab", "Research Lab"],
  ["/data-observatory", "Data Observatory"],
  ["/validation-matrix", "Validation Matrix"],
  ["/defect-register", "Defect Register"],
  ["/model-intelligence", "Model Intelligence"],
  ["/forecast-studio", "Forecast Studio"],
  ["/atmospheric-explorer", "Atmospheric Explorer"],
  ["/reports", "Reports & Reproducibility"],
  ["/settings", "Settings"],
] as const;

test("every module route renders content, not a blank page", async ({ page }) => {
  for (const [path, title] of MODULES) {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: title, level: 2 })).toBeVisible();
    // The main region must contain more than the header.
    const text = await page.locator("main").innerText();
    expect(text.length).toBeGreaterThan(80);
  }
});

test("command center shows live data from the artifact API", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("2,979")).toBeVisible();
  await expect(page.getByText("4.7%").first()).toBeVisible();
});

test("navigation and theme toggle work", async ({ page }) => {
  await page.goto("/");
  const burger = page.getByRole("button", { name: /toggle navigation/i });
  if (await burger.isVisible()) await burger.click();
  await page.getByRole("link", { name: "Defect Register" }).click();
  await expect(page).toHaveURL(/defect-register/);

  const toggle = page.getByRole("button", { name: /theme:/i });
  await toggle.click();
  await toggle.click();
  await toggle.click();
  await expect(toggle).toBeVisible();
});

test("defect search filters the table", async ({ page }) => {
  await page.goto("/defect-register");
  await page.getByRole("searchbox").fill("pressure");
  await expect(page.getByText(/barometric pressure column is corrupt/i)).toBeVisible();
});

test("forecast studio blocks ineligible requests", async ({ page }) => {
  await page.goto("/forecast-studio");
  const select = page.getByLabel(/issue date/i);
  await select.locator("option").nth(1).waitFor({ state: "attached" });
  const value = await select.locator("option").nth(1).getAttribute("value");
  await select.selectOption(value!);
  await page.getByLabel(/^model$/i).selectOption("fusion");
  await expect(page.getByRole("button", { name: /evaluate eligibility/i })).toBeDisabled();
  await expect(page.getByText(/AOD is not observed/i)).toBeVisible();
});
