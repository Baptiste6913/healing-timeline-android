"""Local WiFi server — open the link on your Android/iPhone to test."""
import http.server
import socketserver
import ssl
import os
import socket
import sys
import tempfile
import subprocess

PORT = 4567

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.wasm': 'application/wasm',
        '': 'application/octet-stream',
    }

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def log_message(self, format, *args):
        # Quieter logs
        pass


def get_local_ip():
    """Get the machine's local WiFi IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_self_signed_cert():
    """Generate a temporary self-signed SSL cert for HTTPS (needed for camera access)."""
    cert_dir = tempfile.mkdtemp()
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    try:
        local_ip = get_local_ip()
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_file, "-out", cert_file,
            "-days", "1", "-nodes",
            "-subj", f"/CN={local_ip}",
            "-addext", f"subjectAltName=IP:{local_ip},IP:127.0.0.1",
        ], check=True, capture_output=True)
        return cert_file, key_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


socketserver.TCPServer.allow_reuse_address = True
os.chdir(os.path.dirname(os.path.abspath(__file__)))

local_ip = get_local_ip()

# Try HTTPS first (required for camera access on mobile over network)
cert_file, key_file = generate_self_signed_cert()

if cert_file and key_file:
    httpd = socketserver.TCPServer(("0.0.0.0", PORT), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file, key_file)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print()
    print("=" * 60)
    print("  HEALING TIMELINE — Serveur Local (HTTPS)")
    print("=" * 60)
    print()
    print(f"  Ouvre ce lien sur ton telephone :")
    print()
    print(f"    https://{local_ip}:{PORT}")
    print()
    print(f"  ANDROID (Chrome) :")
    print(f"    → 'Parametres avances' > 'Continuer'")
    print()
    print(f"  iPHONE (Safari) :")
    print(f"    → 'Afficher les details' > 'Visiter le site'")
    print()
    print(f"  (c'est normal, c'est ton propre PC)")
    print(f"  PC et telephone doivent etre sur le meme WiFi !")
    print("=" * 60)
    print()
else:
    httpd = socketserver.TCPServer(("0.0.0.0", PORT), Handler)

    print()
    print("=" * 60)
    print("  HEALING TIMELINE — Serveur Local (HTTP)")
    print("=" * 60)
    print()
    print(f"  Ouvre ce lien sur ton telephone :")
    print()
    print(f"    http://{local_ip}:{PORT}")
    print()
    print(f"  NOTE: La camera ne marchera pas en HTTP.")
    print(f"  Utilise 'Use Sample Mesh (Demo)' a la place.")
    print()
    print(f"  PC et telephone doivent etre sur le meme WiFi !")
    print("=" * 60)
    print()

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    httpd.server_close()
