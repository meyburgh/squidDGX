FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    libwebp-dev libcurl4-openssl-dev ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Makefile .
COPY third_party third_party
COPY cuda cuda
COPY src src
COPY tests tests
# GPU targets: 86=A10G 89=L4/L40S 90=H100 120a=RTX PRO Blackwell 121a=DGX Spark.
# Narrow it for a faster build, e.g. --build-arg CUDA_ARCHS=90
ARG CUDA_ARCHS="86 89 90 120a 121a"
RUN make -j$(nproc) CUDA_ARCHS="$CUDA_ARCHS" all test
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://127.0.0.1:8000/health >/dev/null && curl -fs http://127.0.0.1:8897/health >/dev/null
ENTRYPOINT ["/src/build/squidDGX"]
