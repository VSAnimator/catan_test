import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,  // Using 5174 to avoid conflict with main branch on 5173
  },
  build: {
    outDir: 'dist',
    sourcemap: false, // Disable sourcemaps in production for smaller builds
  },
  // Define environment variables that will be available in the app
  define: {
    // Vite automatically exposes env vars prefixed with VITE_
  },
})

