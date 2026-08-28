import { defineConfig } from 'astro/config';
import fs from 'node:fs';

/**
 * ONE MODE: if a certificate exists, everything is HTTPS. Desktop and phone,
 * identical.
 *
 * There used to be a separate `MOBILE=1` mode, and it was a trap. The backend
 * and the frontend each had to be started in the *same* mode, and whenever
 * they disagreed the page loaded perfectly and then hung forever on
 * "connecting" — because an https:// page may not open a ws:// socket, and a
 * TLS backend will not answer a plaintext one. It looked exactly like the
 * backend being down, and it wasted a lot of time twice.
 *
 * Having a single mode makes that class of failure impossible. HTTPS is the
 * one that has to work, since browsers refuse camera and microphone access on
 * any origin that is not HTTPS or localhost.
 *
 * With no certificate present we fall back to plain HTTP on localhost, which
 * is still a secure context, so a fresh clone works before running any setup.
 */
const certDir = new URL('../certs/', import.meta.url);
const keyPath = new URL('dev-key.pem', certDir);
const certPath = new URL('dev-cert.pem', certDir);

const hasCert = fs.existsSync(keyPath) && fs.existsSync(certPath);

const httpsOptions = hasCert
  ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }
  : undefined;

if (!hasCert) {
  console.warn(
    '[vidi] No certificate found - serving plain HTTP on localhost only.\n' +
      '[vidi] Phones cannot use the camera over http. To enable HTTPS:\n' +
      '[vidi]   backend/.venv/Scripts/python.exe scripts/setup_mobile.py',
  );
}

export default defineConfig({
  output: 'static',
  server: {
    // Bind every interface when we have a certificate, so a phone can reach
    // us; without one there is no point exposing an origin phones cannot use.
    host: hasCert ? true : 'localhost',
    port: 4321,
  },
  vite: {
    server: {
      https: httpsOptions,
      host: hasCert ? true : undefined,
    },
  },
});
