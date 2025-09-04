Local TLS proxy for Weaviate (Cloud-compatible URL)

Goal
- Present your local Weaviate at HTTPS so Elysia can use `connect_to_weaviate_cloud` without code changes.

What it does
- Traefik terminates TLS for `https://weaviate.local` (REST → `weaviate:8080`).
- Traefik also exposes `https://grpc-weaviate.local` (gRPC → `weaviate:50051` via h2c).
- A local CA and server cert are generated; the CA is injected into the Elysia container so Python/httpx/gRPC trust the proxy.

Steps
1) Certificates
   - On startup, the `weaviate-certs` job generates a local CA and a server certificate with SANs:
     - weaviate.local
     - grpc-weaviate.local
   - If certs already exist, they are reused. If SAN for gRPC is missing, they are regenerated.

2) Start services
   - docker compose up -d --build

3) Configure Elysia (UI → Settings)
   - Weaviate Cluster URL: https://weaviate.local
   - API Key: the same one you set in compose for Weaviate (e.g. `elysia-local-admin`)
   - Save

Notes
- The compose mounts `reverse-proxy/certs/ca.crt` and the Elysia entrypoint appends it to certifi and exports:
  - GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/tmp/cacert-plus.pem
  - SSL_CERT_FILE=/tmp/cacert-plus.pem
  - REQUESTS_CA_BUNDLE=/tmp/cacert-plus.pem
- If you also access from your browser, add `weaviate.local` to your host `hosts` file pointing to the Docker host and trust the CA in your OS.
