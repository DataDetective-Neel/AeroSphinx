import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  localStorage.clear();
});

// jsdom does not implement matchMedia, used by the theme provider.
if (!window.matchMedia) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}
// jsdom logs "Not implemented: Window's scrollTo()" for every call. The shell
// resets the scroll position on a route change, so stub it rather than let the
// notice drown out real output.
window.scrollTo = vi.fn();
