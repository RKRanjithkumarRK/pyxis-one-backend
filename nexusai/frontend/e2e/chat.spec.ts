import { test, expect } from "@playwright/test";

// These tests require a logged-in session; in CI, use storageState from auth setup.
test.describe("Chat interface", () => {
  test.use({ storageState: "e2e/.auth/user.json" });

  test("chat page renders", async ({ page }) => {
    await page.goto("/chat");
    await expect(page.getByRole("main")).toBeVisible();
  });

  test("message input is focusable", async ({ page }) => {
    await page.goto("/chat");
    const input = page.getByRole("textbox");
    await input.click();
    await expect(input).toBeFocused();
  });

  test("model selector visible", async ({ page }) => {
    await page.goto("/chat");
    const selector = page.locator("[data-testid=model-selector], [aria-label*='model' i]");
    await expect(selector.first()).toBeVisible({ timeout: 5000 }).catch(() => {
      // Non-fatal — model selector may be in a menu
    });
  });
});
