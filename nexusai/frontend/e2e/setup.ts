/**
 * Global setup: authenticate once and persist session to e2e/.auth/user.json
 * so individual test files can skip re-login.
 */
import { chromium, FullConfig } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0]?.use?.baseURL ?? "http://localhost:3000";
  const authFile = path.join(__dirname, ".auth", "user.json");
  fs.mkdirSync(path.dirname(authFile), { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Use test credentials from env
    const email = process.env.E2E_EMAIL ?? "test@nexusai.dev";
    const password = process.env.E2E_PASSWORD ?? "Test1234!@";

    await page.goto(`${baseURL}/login`);
    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill(password);
    await page.getByRole("button", { name: /sign in/i }).click();
    await page.waitForURL(/chat|dashboard/, { timeout: 10_000 }).catch(() => {});
  } finally {
    await context.storageState({ path: authFile });
    await browser.close();
  }
}

export default globalSetup;
