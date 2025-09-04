#!/bin/sh
set -euo pipefail

OUT_DIR=${OUT_DIR:-/work}

apk add --no-cache openssl >/dev/null 2>&1 || true

# Clean up any mistaken directories created by earlier mounts
for p in ca.crt server.crt server.key server.csr openssl.cnf; do
  if [ -d "$OUT_DIR/$p" ]; then
    echo "Found directory at $OUT_DIR/$p; removing so we can create a file."
    rm -rf "$OUT_DIR/$p"
  fi
done

NEEDS_REGEN=0
if [ ! -f "$OUT_DIR/ca.crt" ] || [ ! -f "$OUT_DIR/server.crt" ]; then
  NEEDS_REGEN=1
else
  # Regenerate if grpc SAN is missing
  if ! openssl x509 -in "$OUT_DIR/server.crt" -text -noout 2>/dev/null | grep -q "DNS:grpc-weaviate.local"; then
    echo "Existing server.crt missing SAN for grpc-weaviate.local; regenerating."
    NEEDS_REGEN=1
  fi
fi

if [ "$NEEDS_REGEN" -eq 0 ]; then
  echo "Certificates already exist in $OUT_DIR with required SANs; skipping generation."
  ls -l "$OUT_DIR"
  exit 0
fi

cat >"$OUT_DIR/openssl.cnf" <<'EOF'
[ req ]
default_bits       = 2048
prompt             = no
default_md         = sha256
x509_extensions    = v3_req
distinguished_name = dn

[ dn ]
CN = weaviate.local

[ v3_req ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = weaviate.local
DNS.2 = grpc-weaviate.local
EOF

openssl genrsa -out "$OUT_DIR/ca.key" 4096
openssl req -x509 -new -nodes -key "$OUT_DIR/ca.key" -sha256 -days 3650 -subj "/CN=Elysia Local CA" -out "$OUT_DIR/ca.crt"
openssl genrsa -out "$OUT_DIR/server.key" 2048
openssl req -new -key "$OUT_DIR/server.key" -out "$OUT_DIR/server.csr" -config "$OUT_DIR/openssl.cnf"
openssl x509 -req -in "$OUT_DIR/server.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial -out "$OUT_DIR/server.crt" -days 825 -sha256 -extensions v3_req -extfile "$OUT_DIR/openssl.cnf"

# Set sane permissions
chmod 600 "$OUT_DIR/ca.key" "$OUT_DIR/server.key"
chmod 644 "$OUT_DIR/ca.crt" "$OUT_DIR/server.crt"

echo "Certificates ready in $OUT_DIR"
ls -l "$OUT_DIR"
