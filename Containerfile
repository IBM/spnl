# Builder
FROM rust:slim AS builder
WORKDIR /tmp/build
COPY Cargo.* .
COPY spnl spnl
COPY cli cli
COPY benchmarks/haystack benchmarks/haystack
COPY web/wasm web/wasm
ARG RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y && apt install -y pkg-config protobuf-compiler sccache libssl-dev
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build -F rag,spnl-api --release
RUN ls -ltrh target/release

# Main
FROM debian:stable-slim
COPY --from=builder /tmp/build/target/release/spnl /usr/local/bin/spnl
RUN groupadd -g 1001 spnl && useradd -u 1001 -g spnl spnl && mkdir /home/spnl && chown spnl /home/spnl
USER spnl

LABEL org.opencontainers.image.source=https://github.com/IBM/spnl
LABEL org.opencontainers.image.description="Span Query CLI"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV SPNL_BUILTIN=email2

ENTRYPOINT ["spnl"]
