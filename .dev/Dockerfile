FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

# Install Node.js and npm (pinned to LTS version 22.x)
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs=22.* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#COPY requirements.in /tmp/requirements.in
COPY requirements.txt /tmp/requirements.txt
USER vscode
ENV PATH=/home/vscode/.local/bin:$PATH

RUN pip install --user -r /tmp/requirements.txt

EXPOSE 10000
