# Running on your phone

The phone is the device this app is really for: it's what you carry, its rear
camera sees the world, and Android Chrome is the only browser here that can
actually vibrate.

---

## The one thing that will trip you up

**Browsers block the camera and microphone on `http://192.168.x.x`.**

`getUserMedia` only works in a *secure context* — HTTPS, or `localhost`. A plain
HTTP address on your LAN is neither, so the page loads but the assistant has no
eyes and no ears. There is no way around this with a plain HTTP server; it is a
browser security rule, not a bug.

So the setup below serves both the frontend and the backend over HTTPS with a
self-signed certificate.

---

## Setup

### 1. Generate the certificate

```bash
backend/.venv/Scripts/python.exe scripts/setup_mobile.py
```

This detects your Wi-Fi address, writes a certificate covering it to `certs/`,
updates `CORS_ORIGINS` in `backend/.env`, and prints the exact run commands.

> **Re-run this whenever your IP changes.** It happens more than you'd think —
> a new DHCP lease or a different network gives you a new address, and the old
> certificate no longer matches. Symptom: everything worked yesterday and today
> the phone can't connect at all. This bit me mid-setup: the address moved from
> `10.165.175.2` to `172.17.208.249` and both servers became unreachable.

### 2. Start the backend (terminal 1)

```bash
cd backend && .venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile ../certs/dev-key.pem --ssl-certfile ../certs/dev-cert.pem
```

`--host 0.0.0.0` matters: bound to `127.0.0.1` it is invisible to the phone.

### 3. Start the frontend (terminal 2)

```bash
cd frontend && npm run dev:mobile
```

### 4. On the phone — **do these in order**

**Step 1: trust the backend certificate first.** Open

```
https://<your-ip>:8000/health
```

Tap **Advanced → Proceed**. You should see JSON.

This step is easy to skip and nothing works without it. The app page will load
fine, but every WebSocket silently fails — browsers refuse an untrusted
certificate for `wss://` with **no prompt at all**, so you get a page that looks
right and does nothing.

**Step 2: open the app.**

```
https://<your-ip>:4321
```

Accept the warning again, tap **Start assistant**, allow camera and microphone.

---

## What's different on a phone

| | |
|---|---|
| **Camera** | Uses the rear camera (`facingMode: environment`) — it should look at the world, not your face |
| **Haptics** | `navigator.vibrate` works, so patterns genuinely fire. One motor, so rhythm conveys urgency but **not** direction — the system says so rather than implying otherwise |
| **Layout** | Controls fixed at the bottom in thumb reach; camera and conversation first; the eight developer panels collapse to one tappable line each |
| **Touch** | Every target ≥ 44px. Inputs are 16px so iOS doesn't zoom on focus |
| **Notch** | Safe-area insets keep the control bar clear of the home indicator |

Verified at 375×812: no horizontal overflow, controls fixed at `bottom: 0`,
48px start button, zero touch targets under 44px.

---

## How the backend address is found

Nothing is hardcoded. The page derives it from where *it* was loaded from:

```js
const secure = window.location.protocol === 'https:';
return (secure ? 'wss://' : 'ws://') + window.location.hostname + ':8000/ws';
```

A hardcoded `127.0.0.1` would mean **the phone itself**. The scheme has to match
too — an HTTPS page cannot open an insecure `ws://` socket; browsers block it as
mixed content.

Override with `PUBLIC_BACKEND_WS_URL` in `frontend/.env` only if the backend
lives on a different machine entirely.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Page loads, status stuck "Backend unreachable" | Certificate not accepted for port **8000**. Do step 1 |
| "Could not start: Permission denied" | Camera/mic denied, or you're on `http://` not `https://` |
| Phone can't reach either address at all | Not on the same Wi-Fi, IP changed (re-run the script), or client isolation is on — common on guest and campus networks |
| Worked yesterday, dead today | Your IP changed. Re-run `setup_mobile.py` |
| No vibration | You're on desktop, or iOS Safari (no vibration API). Android Chrome only |
| Works but sluggish | The phone is only capturing and playing audio; all inference runs on the laptop. Check both are on 5GHz Wi-Fi |

---

## Security

The certificate is self-signed and **for local development only**. `certs/` is
gitignored — the private key must never be committed. Anyone on your Wi-Fi can
reach the app while it's running, and there is no authentication, so don't leave
it running on a public network.
