FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV Torch_DIR=/usr/local/lib/libtorch
WORKDIR /workspace

RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \
    build-essential \
    wget \
    gnupg \
    lsb-release \
    software-properties-common && \
    echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6A755776 && \
    apt update && \
    apt install -y --no-install-recommends \
    unzip \
    sudo \
    cmake \
    nano \
    git \
    graphviz \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libboost-all-dev \
    libboost-filesystem-dev \
    nlohmann-json3-dev \
    nvidia-cuda-toolkit \   
    libzmq3-dev \
    libczmq-dev \
    cppzmq-dev \
    libmsgpack-dev \
    catch2 \
    screen \
    gdb \
    libtbb-dev \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -O libtorch.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip && \
    mv libtorch /usr/local/lib/

RUN python3.12 -m venv /opt/gprm_venv
ENV PYTHON_INTERPRETER_PATH="/opt/gprm_venv/bin/python3.12"
ENV PATH="/opt/gprm_venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
