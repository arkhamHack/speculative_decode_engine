/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        gpu: {
          green:  '#22c55e',
          teal:   '#14b8a6',
          blue:   '#3b82f6',
          purple: '#a855f7',
          orange: '#f97316',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
