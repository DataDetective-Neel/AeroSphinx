export const COLORS = {
  // Dark palette
  dark: {
    bg: "#0B1118",
    card: "#131D28",
    cardHover: "#1A2633",
    text: "#E8EEF5",
    muted: "#9AAABC",
    accent: "#38BDF8",
    success: "#22C55E",
    warning: "#F59E0B",
    error: "#EF4444",
    border: "#293747",
  },
  // Light palette
  light: {
    bg: "#F5F7FA",
    card: "#FFFFFF",
    cardHover: "#EDF2F7",
    text: "#172334",
    muted: "#64748B",
    accent: "#0284C7",
    success: "#15803D",
    warning: "#B45309",
    error: "#B91C1C",
    border: "#DCE3EB",
  },
  // System
  system: "system",
} as const

export const TYPOGRAPHY = {
  fontSans: "\"Geist Sans\", system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif",
  fontMono: "\"Geist Mono\", ui-monospace, SF Mono, Menlo, Monaco, Consolas, monospace",
  fontSize: {
    xs: "0.75rem",
    sm: "0.875rem",
    base: "1rem",
    lg: "1.125rem",
    xl: "1.25rem",
    "2xl": "1.5rem",
    "3xl": "1.875rem",
    "4xl": "2.25rem",
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
} as const
