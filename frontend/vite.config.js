import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const projectRoot = fileURLToPath(new URL('..', import.meta.url));
const frontendRoot = fileURLToPath(new URL('.', import.meta.url));
const metricsPath = path.join(projectRoot, 'results', 'metrics.csv');
const toyPath = path.join(projectRoot, 'data', 'demo', 'toy.json');
const backendTarget = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8001';

// Dev always reads the newest batch results; a static build includes a snapshot.
function resultsAsset() {
  return {
    name: 'topicsum-results',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url?.split('?')[0] !== '/experiment/metrics.csv') return next();
        response.setHeader('Content-Type', 'text/csv; charset=utf-8');
        response.setHeader('Cache-Control', 'no-store');
        response.end(existsSync(metricsPath) ? readFileSync(metricsPath) : '');
      });
    },
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'experiment/metrics.csv',
        source: existsSync(metricsPath) ? readFileSync(metricsPath) : '',
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), resultsAsset()],
  server: {
    fs: { allow: [frontendRoot, toyPath] },
    proxy: {
      '/api': backendTarget,
      '/health': backendTarget,
    },
  },
  preview: {
    proxy: {
      '/api': backendTarget,
      '/health': backendTarget,
    },
  },
});
