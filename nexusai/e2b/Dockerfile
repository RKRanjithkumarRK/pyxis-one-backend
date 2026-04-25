# NexusAI E2B template — Ubuntu 22.04 + all toolchains
# Build with: e2b template build --name nexusai-dev
FROM e2bdev/code-interpreter:latest

# System packages
RUN apt-get update && apt-get install -y \
    curl wget git vim nano jq \
    build-essential cmake \
    python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm yarn bun tsx \
    && rm -rf /var/lib/apt/lists/*

# Go 1.22
RUN wget -q https://go.dev/dl/go1.22.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz \
    && rm go1.22.0.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=$PATH:/root/.cargo/bin

# Java 21 (Temurin)
RUN wget -q https://packages.adoptium.net/artifactory/deb/pool/main/t/temurin-21/temurin-21-jdk_21.0.3+9_amd64.deb 2>/dev/null || true

# Python: common packages
RUN pip3 install --no-cache-dir \
    fastapi uvicorn httpx requests pandas numpy scipy matplotlib \
    openai anthropic transformers torch --index-url https://download.pytorch.org/whl/cpu \
    jupyter ipython

# Ruby & PHP
RUN apt-get update && apt-get install -y ruby-full php-cli && rm -rf /var/lib/apt/lists/*

# .NET 8
RUN wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x dotnet-install.sh \
    && ./dotnet-install.sh --channel 8.0 --install-dir /usr/local/dotnet \
    && rm dotnet-install.sh
ENV PATH=$PATH:/usr/local/dotnet

# Set default workdir
WORKDIR /workspace

# Default shell
CMD ["bash"]
