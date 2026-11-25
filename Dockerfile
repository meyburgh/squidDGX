FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS engine
RUN apt-get update && apt-get install -y --no-install-recommends libwebp-dev libcurl4-openssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Makefile .
COPY third_party third_party
COPY cuda cuda
COPY src src
COPY tests tests
RUN make -j$(nproc) all test
ENTRYPOINT ["/src/build/squidDGX"]
