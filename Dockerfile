# FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ubuntu:22.04

WORKDIR /opt/ml

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install gcc -y \
  && apt-get install zip -y \
  && apt-get install unzip -y \
  && apt-get install python3.10 -y \
  && apt-get install python3-pip -y \
  && apt-get install curl -y \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && apt-get update \
  && apt-get install nvidia-container-toolkit -y \
  && apt-get clean

COPY requirements.txt /opt/ml/requirements.txt

RUN pip install -r requirements.txt

COPY debarta_training/. /opt/ml/debarta_training/
COPY dpr_training/. /opt/ml/dpr_training/
