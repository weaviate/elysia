import type { Config } from "tailwindcss";

const goldenRatio = 1.518;
const baseFontSize = 1; // 1rem (16px)
const fontSizes: { [key: string]: string } = {};

for (let i = -3; i <= 6; i++) {
  const size = baseFontSize * Math.pow(goldenRatio, i);
  const roundedSize = size.toFixed(3);
  let name;
  if (i < 0) {
    name = `${-i}xs`; // '1xs', '2xs', '3xs'
  } else if (i === 0) {
    name = "base";
  } else {
    name = i === 1 ? "xl" : `${i}xl`; // 'xl', '2xl', etc.
  }
  fontSizes[name] = `${roundedSize}rem`;
}

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        outfit: ["var(--font-outfit)"],
        merriweather: ["var(--font-merriweather)"],
      },
      colors: {
        background: "#2D1826",
        background_alt: "#371C2E",
        foreground: "#3B2639",
        foreground_alt: "#45273B",
        primary: "#FFF1E4",
        secondary: "#A07091",
        accent: "#FFBC74",
        error: "#BF4242",
        background_secondary: "#986E83",
        background_accent: "#D69959",
        background_highlight: "#D4A87A",
        background_error: "#993434",
        highlight: "#FFD6AB",
        warning: "#DE774B",
      },
      fontSize: fontSizes,
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
