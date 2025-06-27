FROM ubuntu:22.04

RUN apt update
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    nano \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx unzip \
    curl cron ffmpeg \
    ssh openssh-client \
    vim \
    git

WORKDIR /work
ENV PYTHONPATH="/work"

RUN pip install uv
RUN uv init
RUN uv sync

CMD ["bash"]
