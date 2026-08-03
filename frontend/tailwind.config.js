/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#1f1d1d',
          light: '#dde1e4',
          teal: '#9ac4c6',
          'teal-dark': '#307784',
        }
      }
    },
  },
  plugins: [],
}
