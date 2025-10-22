#!/usr/bin/env bash
set -ex

echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/nordquant/ai-agents-crash-course-codespace:latest-release . --push

