# syntax=docker/dockerfile:1
FROM golang:1.24-trixie AS builder

WORKDIR /build

# Download onnxruntime shared library.
ARG ORT_VERSION=1.24.1
ADD https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz /tmp/ort.tgz
RUN tar -xzf /tmp/ort.tgz -C /tmp && \
    cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION} /usr/local/lib/ && \
    ln -sf libonnxruntime.so.${ORT_VERSION} /usr/local/lib/libonnxruntime.so && \
    ldconfig

# Cache Go modules.
COPY go.mod go.sum ./
RUN go mod download

# Build the binary.
COPY . .
RUN go build -o /infer ./cmd/infer/

# Runtime image.
FROM debian:trixie-slim

ARG ORT_VERSION=1.24.1
COPY --from=builder /usr/local/lib/libonnxruntime.so.${ORT_VERSION} /usr/local/lib/
RUN ln -sf libonnxruntime.so.${ORT_VERSION} /usr/local/lib/libonnxruntime.so && ldconfig

COPY --from=builder /infer /usr/local/bin/infer

RUN useradd -m -u 1000 infer
USER infer

WORKDIR /app
ENTRYPOINT ["infer"]
