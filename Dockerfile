# Use the official Jenkins LTS image as base
FROM jenkins/jenkins:lts

# Switch to root to install extra packages
USER root

# Install dependencies (git, ssh, etc.)
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set Jenkins to run with Jenkins user
USER jenkins

# Expose Jenkins port
EXPOSE 8080

# Expose agent port (optional, for Jenkins agents)
EXPOSE 50000

# Default Jenkins entrypoint (already included)
