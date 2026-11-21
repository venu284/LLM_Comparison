# S1.7 checkpoint 2b toolchain image (Amendment 9). Base: the python:3.10-slim digest of the
# DS-1000 image (the sandbox boot program needs python3). Network only at build time; cells run
# with --network none. Built for linux/arm64; referenced by local image id (IMAGE_ID).
# CodeGeeX HumanEval-X C++: g++ -std=c++11 (+ -lcrypto -lssl for CPP/162), boost and OpenSSL
# headers (CodeGeeX's Docker image installs Boost 1.71 and OpenSSL 3.0; Debian bookworm's are used).
FROM python:3.10-slim@sha256:6f16c98fd47a233f0195fb4e683539a86211c2cea0ba448966612b38789596c9
RUN apt-get update && apt-get install -y --no-install-recommends g++ libboost-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/* && dpkg-query -W g++ libboost-dev libssl-dev > /opt/toolchain-versions.txt
