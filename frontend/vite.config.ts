import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // バックエンド (FastAPI) へ /api を中継する
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
