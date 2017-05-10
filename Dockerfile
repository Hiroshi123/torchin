FROM ubuntu:14.04

# Install
RUN \
    sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y build-essential clang-3.5 && \
    apt-get install -y software-properties-common && \
    apt-get install -y byobu curl git htop man unzip vim wget

RUN \
    apt-get install -y subversion cmake qt5-default && \
    rm -rf /var/lib/apt/lists/*
    
