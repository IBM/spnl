import { test, expect } from '@playwright/test';

test('has title', async ({ page }) => {
  await page.goto('http://localhost:5173/');

  // Expect a title
  await expect(page).toHaveTitle(/Span Queries/);
});

test('query editor open with no compilation errors', async ({ page }) => {
  await page.goto('http://localhost:5173/');
  await expect(page.getByRole('textbox', { name: 'Query Editor' })).toHaveCount(1)
  await expect(page.getByRole('listitem', { name: 'Query compilation error' })).toHaveCount(0)
});

test('click on toggle query viewer button', async ({ page }) => {
  await page.goto('http://localhost:5173/');
  await page.getByRole('button', { name: 'Toggle Query Viewer' }).click();

  // hmm doesn't work: await expect(page.getByRole('generic', { name: 'Query Viewer' })).toHaveCount(1);
  await expect(page.locator('[aria-label="Query Viewer"]')).toHaveCount(1);
})
