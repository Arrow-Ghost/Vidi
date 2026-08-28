"""
Set this app up to run on a phone over your Wi-Fi.

WHY THIS SCRIPT EXISTS

Browsers only expose the camera and microphone in a "secure context": HTTPS,
or localhost. Opening the dashboard at http://192.168.x.x:4321 from a phone is
NEITHER, so getUserMedia is blocked outright and the assistant has no eyes and
no ears. There is no way around that with a plain HTTP LAN server.

So this generates a self-signed certificate covering this machine's LAN IP and
prints the exact commands to serve both the frontend and the backend over
HTTPS/WSS. The phone will warn once that the certificate is not trusted (it is
signed by nobody) — accepting it is what makes the camera work.

Run:  python scripts/setup_mobile.py
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERT_DIR = ROOT / "certs"
CERT_FILE = CERT_DIR / "dev-cert.pem"
KEY_FILE = CERT_DIR / "dev-key.pem"

FRONTEND_PORT = 4321
BACKEND_PORT = 8000


def detect_lan_ip() -> str:
    """
    The address a phone on the same Wi-Fi can actually reach.

    Uses a UDP socket to a public address: no packet is sent, but the OS picks
    the interface it WOULD route through, which is the real Wi-Fi address
    rather than a VMware/Hyper-V virtual adapter. Enumerating adapters instead
    reliably picks the wrong one on machines with virtualization installed.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def all_candidate_ips() -> list:
    """
    EVERY IPv4 address on this machine.

    Deliberately thorough. DHCP hands out a new address whenever you rejoin a
    network, and a certificate that covers only the address seen at generation
    time silently stops working — the phone reports "cannot connect" with no
    hint that the cause is a stale SAN list. Covering every interface means one
    certificate keeps working across reconnects and network changes.
    """
    ips = set()
    try:
        import psutil

        for addresses in psutil.net_if_addrs().values():
            for address in addresses:
                if getattr(address, "family", None) == socket.AF_INET:
                    ips.add(address.address)
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass

    ips.add(detect_lan_ip())
    # Link-local 169.254.x is never reachable from a phone; keep the list clean.
    return sorted(
        i for i in ips
        if not i.startswith("127.") and not i.startswith("169.254.")
    )


def generate_cert(lan_ip: str, extra_ips: list) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # The SAN list is the part that matters: browsers ignore CN entirely, so a
    # cert without the IP in subjectAltName is rejected even when you accept
    # the warning.
    alt_names = [x509.DNSName("localhost")]
    seen = set()
    for address in [lan_ip, "127.0.0.1", *extra_ips]:
        if address in seen:
            continue
        seen.add(address)
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(address)))
        except ValueError:
            continue

    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Vidi Assistive Agent (dev)"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def update_cors(lan_ip: str) -> None:
    """Allow the phone's origin, or the backend will reject its requests."""
    env_path = ROOT / "backend" / ".env"
    if not env_path.exists():
        return
    origins = [
        "http://localhost:" + str(FRONTEND_PORT),
        "http://127.0.0.1:" + str(FRONTEND_PORT),
        "https://localhost:" + str(FRONTEND_PORT),
        "https://127.0.0.1:" + str(FRONTEND_PORT),
        "https://" + lan_ip + ":" + str(FRONTEND_PORT),
        "http://" + lan_ip + ":" + str(FRONTEND_PORT),
    ]
    lines = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("CORS_ORIGINS="):
            line = "CORS_ORIGINS=" + ",".join(dict.fromkeys(origins))
        lines.append(line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    lan_ip = detect_lan_ip()
    extras = all_candidate_ips()

    if lan_ip.startswith("127."):
        print("Could not find a LAN address. Is Wi-Fi connected?")
        return 1

    print("Detected LAN address : " + lan_ip + "   <- use this one on the phone")
    others = [a for a in extras if a != lan_ip]
    if others:
        print("Also covered by cert : " + ", ".join(others))
        print("  (virtual/secondary adapters - included so the certificate keeps")
        print("   working if your address changes, but not what the phone uses)")

    try:
        generate_cert(lan_ip, extras)
    except ImportError:
        print("\nThe 'cryptography' package is required:")
        print("  backend/.venv/Scripts/python.exe -m pip install cryptography")
        return 1

    update_cors(lan_ip)

    print("\nCertificate written to " + str(CERT_DIR))
    print("CORS updated in backend/.env\n")

    bar = "=" * 68
    print(bar)
    print("RUN THESE, ONE PER TERMINAL")
    print(bar)
    print("\n1) Backend (HTTPS + WSS, reachable on the LAN):\n")
    print("   cd backend")
    print("   .venv/Scripts/python.exe -m uvicorn app.main:app \\")
    print("       --host 0.0.0.0 --port " + str(BACKEND_PORT) + " \\")
    print("       --ssl-keyfile ../certs/dev-key.pem \\")
    print("       --ssl-certfile ../certs/dev-cert.pem")
    print("\n2) Frontend (HTTPS, reachable on the LAN):\n")
    print("   cd frontend")
    print("   npm run dev:mobile")
    print("\n" + bar)
    print("ON YOUR PHONE (same Wi-Fi)")
    print(bar)
    print("\n   STEP 1 - trust the backend certificate FIRST. Open:")
    print("       https://" + lan_ip + ":" + str(BACKEND_PORT) + "/health")
    print("     Tap Advanced -> Proceed. You should see JSON.")
    print("     This step is easy to skip and the app will not work without it:")
    print("     the page loads but every WebSocket silently fails, because the")
    print("     browser refuses the untrusted cert for wss:// with no prompt.")
    print("\n   STEP 2 - open the app:")
    print("       https://" + lan_ip + ":" + str(FRONTEND_PORT))
    print("     Accept the warning again, then tap 'Start assistant' and allow")
    print("     camera and microphone.")
    print("\n   Android Chrome also gives you REAL vibration - the haptic")
    print("   patterns actually fire on a phone.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
