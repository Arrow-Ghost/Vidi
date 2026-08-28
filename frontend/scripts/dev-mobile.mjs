/**
 * Start the Astro dev server in "phone" mode: HTTPS, bound to every
 * interface, using the certificate from scripts/setup_mobile.py.
 *
 * A script file rather than an inline `node -e` one-liner in package.json,
 * and rather than `MOBILE=1 astro dev` which is not portable to PowerShell
 * or cmd. astro.config.mjs reads process.env.MOBILE to decide whether to
 * attach the certificate.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const certDir = path.resolve(here, '..', '..', 'certs');
const key = path.join(certDir, 'dev-key.pem');
const cert = path.join(certDir, 'dev-cert.pem');

if (!fs.existsSync(key) || !fs.existsSync(cert)) {
  console.error('\nNo certificate found in ' + certDir);
  console.error('Generate one first:\n');
  console.error('  backend/.venv/Scripts/python.exe scripts/setup_mobile.py\n');
  process.exit(1);
}

const child = spawn('npx', ['astro', 'dev', '--host'], {
  stdio: 'inherit',
  shell: true,
  env: { ...process.env, MOBILE: '1' },
});
child.on('exit', (code) => process.exit(code ?? 0));
