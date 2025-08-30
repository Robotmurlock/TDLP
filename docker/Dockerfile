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

RUN apt-get install -y git python3.10-venv
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX /YOLOX
WORKDIR /YOLOX
RUN python3 -m venv venv
RUN /bin/bash -c "source /YOLOX/venv/bin/activate && pip install -U pip setuptools"
RUN /bin/bash -c "source /YOLOX/venv/bin/activate && pip install torch torchvision torchaudio"
RUN /bin/bash -c "source /YOLOX/venv/bin/activate && pip install --no-build-isolation -v -e ."

WORKDIR /work

CMD ["bash"]
