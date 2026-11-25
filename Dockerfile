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
RUN make -j$(nproc) all test
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://127.0.0.1:8000/health >/dev/null && curl -fs http://127.0.0.1:8897/health >/dev/null
ENTRYPOINT ["/src/build/squidDGX"]
