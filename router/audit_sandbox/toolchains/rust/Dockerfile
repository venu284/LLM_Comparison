# S1.7 checkpoint 2b toolchain image (Amendment 9). Base: the python:3.10-slim digest of the
# DS-1000 image (the sandbox boot program needs python3). Network only at build time; cells run
# with --network none. Built for linux/arm64; referenced by local image id (IMAGE_ID).
# MultiPL-E Rust: rustc <file> -o <bin> (no edition flag), then run the binary.
FROM rust:1.79.0-slim-bookworm@sha256:073c8e7ae12d2637d1d541cb0f3f9d92bfa22c97e2e45db87a9fc6a90c9a7f22 AS rust
FROM python:3.10-slim@sha256:6f16c98fd47a233f0195fb4e683539a86211c2cea0ba448966612b38789596c9
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY --from=rust /usr/local/cargo /usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN rustc --version > /opt/toolchain-versions.txt && chmod -R a+rX /usr/local/rustup /usr/local/cargo
