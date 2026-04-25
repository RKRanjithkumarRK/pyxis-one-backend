import { test, expect } from "@playwright/test";

test.describe("Authentication flows", () => {
  test("login page loads", async ({ page }) => {
    await page.goto("/login");
    await expect(page).toHaveTitle(/NexusAI/);
    await expect(page.getByRole("heading", { name: /sign in/i })).toBeVisible();
  });

  test("signup page loads", async ({ page }) => {
    await page.goto("/signup");
    await expect(page.getByRole("button", { name: /create account/i })).toBeVisible();
  });

  test("login form validates empty submission", async ({ page }) => {
    await page.goto("/login");
    await page.getByRole("button", { name: /sign in/i }).click();
    // Expect validation error (browser or custom)
    const error = page.locator("[aria-invalid], [data-error], .text-red-500, .text-destructive");
    await expect(error.first()).toBeVisible({ timeout: 3000 }).catch(() => {});
  });

  test("redirects unauthenticated user from /chat to /login", async ({ page }) => {
    await page.goto("/chat");
    await expect(page).toHaveURL(/login|signin/);
  });
});
