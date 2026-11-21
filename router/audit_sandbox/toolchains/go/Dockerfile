# S1.7 checkpoint 2b toolchain image (Amendment 9). Base: the python:3.10-slim digest of the
# DS-1000 image (the sandbox boot program needs python3). Network only at build time; cells run
# with --network none. Built for linux/arm64; referenced by local image id (IMAGE_ID).
# CodeGeeX HumanEval-X Go: `go test main_test.go` with testify. testify is vendored at build
# time into /opt/goeval (go.mod + vendor/); each cell copies it to /work and runs with
# GOFLAGS=-mod=vendor GOPROXY=off.
FROM golang:1.22.5-bookworm@sha256:af9b40f2b1851be993763b85288f8434af87b5678af04355b1e33ff530b5765f AS go
FROM python:3.10-slim@sha256:6f16c98fd47a233f0195fb4e683539a86211c2cea0ba448966612b38789596c9
COPY --from=go /usr/local/go /usr/local/go
ENV PATH=/usr/local/go/bin:$PATH GOPATH=/opt/gopath GOCACHE=/opt/gocache GOTOOLCHAIN=local CGO_ENABLED=0
RUN mkdir -p /opt/goeval && cd /opt/goeval \
    && printf 'module goeval\n\ngo 1.22\n\nrequire github.com/stretchr/testify v1.9.0\n' > go.mod \
    && printf 'package main\n\nimport (\n\t"testing"\n\t"github.com/stretchr/testify/assert"\n)\n\nfunc TestX(t *testing.T) { assert.New(t).Equal(1, 1) }\n' > main_test.go \
    && go mod tidy && go mod vendor && go test -mod=vendor ./... && rm main_test.go \
    && rm -rf /opt/gocache /opt/gopath && chmod -R a+rX /opt/goeval && go version > /opt/toolchain-versions.txt
