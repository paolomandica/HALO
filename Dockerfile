FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Set timezone
ENV TZ=Europe/Rome
RUN ln -sf /usr/share/zoneinfo/Europe/Rome /etc/localtime

# Update the package list and install dependencies
RUN apt-get update && apt-get install -y \
    dialog \
    apt-utils \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    nano \
    tmux \
    screen \
    tree \
    htop \
    zip \
    unzip \
    ca-certificates \
    sudo

# install python and pip
RUN apt-get install -y python3 python3-pip
# set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# copy and install pip requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create a non-root user
ARG UID
ARG GID
RUN addgroup --gid $GID nonroot && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" nonroot
RUN echo "nonroot ALL=(ALL) ALL" >> /etc/sudoers

# Set the default user to the non-root user
USER nonroot

# Set the working directory
WORKDIR /home/nonroot
