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
        background: "#061026",
        background_alt: "#09122F",
        foreground: "#0B173A",
        foreground_alt: "#0D1B46",
        primary: "#ffffff",
        secondary: "#464C5F",
        accent: "#5AFF83",
        error: "#BA2B2B",
        background_secondary: "#1F5A91",
        background_accent: "#308755",
        background_error: "#601E29",
        highlight: "#FF72FF",
        warning: "#D75923",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
export default config;
