import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#050712",
        panel: "rgba(13, 22, 42, 0.72)",
        navy: "#08111f",
        electric: "#2f7cff",
        purple: "#8b5cf6",
        emerald: "#10b981",
        danger: "#ef4444",
      },
      boxShadow: {
        glow: "0 0 40px rgba(47, 124, 255, 0.26)",
        emerald: "0 0 30px rgba(16, 185, 129, 0.25)",
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
} satisfies Config;
