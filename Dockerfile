FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS engine
WORKDIR /src
COPY Makefile .
COPY cuda cuda
COPY src src
COPY tests tests
RUN make -j$(nproc) all test
ENTRYPOINT ["/src/build/squidDGX"]
