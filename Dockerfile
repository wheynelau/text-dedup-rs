FROM python:3.10-slim

RUN apt-get update && apt-get install -y git gcc curl openjdk-17-jdk openjdk-17-jre-headless && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSfcurl https://sh.rustup.rs -sSf | sh -s -- -y

WORKDIR /app
RUN git clone https://github.com/google-research/deduplicate-text-datasets.git
WORKDIR /app/deduplicate-text-datasets
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo build

WORKDIR /app

COPY . /app

RUN pip install .

ENTRYPOINT ["/bin/bash", "-c"]
