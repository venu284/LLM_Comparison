# S1.7 audit: DS-1000 execution environment, from DS-1000's own environment.yml
# (github.com/xlang-ai/DS-1000 @ cddd24520f374c6cdf3497adf058467394f3cf0b).
# Base pinned by digest (python:3.10-slim, linux/arm64 v8). Network is used only at build time;
# cells run with --network none.
# Deviation: DS-1000 pins tensorflow-cpu==2.16.1, which has no linux/aarch64 wheel; the CPU-only
# aarch64 build of tensorflow==2.16.1 is installed instead. torch 2.2.0 aarch64 wheels are CPU-only.
FROM python:3.10-slim@sha256:6f16c98fd47a233f0195fb4e683539a86211c2cea0ba448966612b38789596c9
RUN pip install --no-cache-dir \
    gensim==4.3.2 matplotlib==3.8.4 numpy==1.26.4 pandas==1.5.3 seaborn==0.13.2 \
    scikit-learn==1.4.0 scipy==1.12.0 statsmodels==0.14.1 xgboost==2.0.3 pyyaml \
    torch==2.2.0 tensorflow==2.16.1
ENV TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1
