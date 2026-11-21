# S1.7 checkpoint 2b toolchain image (Amendment 9). Base: the python:3.10-slim digest of the
# DS-1000 image (the sandbox boot program needs python3). Network only at build time; cells run
# with --network none. Built for linux/arm64; referenced by local image id (IMAGE_ID).
# CodeGeeX HumanEval-X JavaScript (node test.js) and MultiPL-E TypeScript (tsc --target esnext,
# then node).
FROM node:20.15.1-bookworm-slim@sha256:b21bcf3e7b6e68d723eabedc6067974950941167b5d7a9e414bd5ac2011cd1c4 AS node
FROM python:3.10-slim@sha256:6f16c98fd47a233f0195fb4e683539a86211c2cea0ba448966612b38789596c9
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g --no-fund --no-audit typescript@5.4.5 && rm -rf /root/.npm \
    && (node --version; tsc --version) > /opt/toolchain-versions.txt
