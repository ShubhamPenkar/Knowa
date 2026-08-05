/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // KNOWA dark system (aligned with HalftoneGlow accents)
        ink: '#F2EEE6',       // primary text on dark
        paper: '#0A0908',     // app ground
        mist: '#2A2622',      // borders / rails
        surface: '#141210',   // elevated panels
        muted: {
          DEFAULT: '#9A948A',
        },
        teal: {
          DEFAULT: '#00C8B4',
          soft: '#0A3330',
        },
        coral: {
          DEFAULT: '#FF5A1F',
          soft: '#3A1A10',
        },
        blue: {
          DEFAULT: '#4A7FBE',
          soft: '#152033',
        },
        brand: {
          dark: '#0A0908',
          light: '#141210',
          teal: '#00C8B4',
          'teal-dark': '#00A898',
          ink: '#F2EEE6',
          paper: '#0A0908',
          mist: '#2A2622',
          sea: '#00C8B4',
          seaMid: '#00C8B4',
          seaSoft: '#0A3330',
          seafoam: '#0A3330',
          copper: '#FF5A1F',
          muted: '#9A948A',
        },
      },
      fontFamily: {
        display: ['Syne', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['"DM Sans"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        control: '6px',
      },
      maxWidth: {
        content: '72rem',
      },
      keyframes: {
        'spine-in': {
          '0%': { transform: 'scaleX(0)', opacity: '0.4' },
          '100%': { transform: 'scaleX(1)', opacity: '1' },
        },
        'page-in': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'spine-in': 'spine-in 400ms ease-out both',
        'page-in': 'page-in 250ms ease-out both',
      },
    },
  },
  plugins: [],
}
