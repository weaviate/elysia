import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#121212",
        background_alt: "#171717",
        foreground: "#1E1E1E",
        foreground_alt: "#262626",
        primary: "#ffffff",
        secondary: "#4E4E4E",
        accent: "#4EFF77",
        error: "#BA2B2B",
        background_secondary: "#343434",
        background_accent: "#10932E",
        background_highlight: "#A60D71",
        background_error: "#601E29",
        highlight: "#FF17AE",
        warning: "#D75923",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
export default config;
