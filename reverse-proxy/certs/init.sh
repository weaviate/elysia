#!/bin/sh
set -eu

# Working dir is the mounted certs folder
cd /work

WEAVIATE_HOST_1="weaviate.local"
WEAVIATE_HOST_2="grpc-weaviate.local"

need_gen() {
  # If any required file is missing, we need to (re)generate
  [ -f ca.crt ] && [ -f ca.key ] && [ -f server.crt ] && [ -f server.key ] || return 0
  # Verify SANs exist; if not, regenerate
  if ! openssl x509 -in server.crt -noout -text 2>/dev/null | grep -q "DNS:${WEAVIATE_HOST_1}"; then
    return 0
  fi
  if ! openssl x509 -in server.crt -noout -text 2>/dev/null | grep -q "DNS:${WEAVIATE_HOST_2}"; then
    return 0
  fi
  # Everything looks good; no need to regenerate
  return 1
}

echo "[weaviate-certs] Preparing local CA and server certificate..."
if need_gen; then
  echo "[weaviate-certs] Generating new local CA and server certs"

  # Create a minimal OpenSSL config for SAN entries
  cat > /tmp/openssl.cnf <<EOF
[ req ]
default_bits       = 2048
distinguished_name = req_distinguished_name
req_extensions     = v3_req
prompt             = no

[ req_distinguished_name ]
CN = ${WEAVIATE_HOST_1}

[ v3_req ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = ${WEAVIATE_HOST_1}
DNS.2 = ${WEAVIATE_HOST_2}
EOF

  # Generate CA key and certificate (10 years)
  openssl genrsa -out ca.key 4096 >/dev/null 2>&1
  openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -subj "/CN=Local Dev CA" -out ca.crt >/dev/null 2>&1

  # Generate server key and CSR with SANs
  openssl genrsa -out server.key 2048 >/dev/null 2>&1
  openssl req -new -key server.key -out server.csr -config /tmp/openssl.cnf >/dev/null 2>&1

  # Sign server cert with our CA, including SANs
  openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -days 3650 -sha256 -extensions v3_req -extfile /tmp/openssl.cnf >/dev/null 2>&1

  # Clean up CSR and temp files
  rm -f server.csr /tmp/openssl.cnf || true
else
  echo "[weaviate-certs] Existing certs found with required SANs; skipping regeneration"
fi

echo "[weaviate-certs] Done"

