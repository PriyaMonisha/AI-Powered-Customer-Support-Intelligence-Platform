# filename: scripts/generate_env.py
# purpose:  One-time local setup — copy .env.example to .env with real generated
#           Fernet/secret keys for Airflow (Section 14). Never overwrites an
#           existing .env. Stdlib-only (cryptography is not in requirements.txt).
import base64
import os
import secrets
from pathlib import Path

src = Path(__file__).resolve().parent.parent / ".env.example"
dst = Path(__file__).resolve().parent.parent / ".env"

if dst.exists():
    print(".env already exists — not overwriting")
else:
    fernet_key = base64.urlsafe_b64encode(os.urandom(32)).decode()  # == Fernet.generate_key()
    secret_key = secrets.token_hex(16)

    content = src.read_text()
    content = content.replace("AIRFLOW__CORE__FERNET_KEY=", f"AIRFLOW__CORE__FERNET_KEY={fernet_key}")
    content = content.replace("AIRFLOW__WEBSERVER__SECRET_KEY=", f"AIRFLOW__WEBSERVER__SECRET_KEY={secret_key}")

    assert f"AIRFLOW__CORE__FERNET_KEY={fernet_key}" in content, "Fernet key replacement failed — check .env.example format"
    assert f"AIRFLOW__WEBSERVER__SECRET_KEY={secret_key}" in content, "Secret key replacement failed — check .env.example format"

    dst.write_text(content)
    print(".env generated with Fernet key + secret key")
