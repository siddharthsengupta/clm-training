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
  && apt-get install nvidia-utils-550-server -y \
  && apt-get clean

COPY requirements.txt /opt/ml/requirements.txt

RUN pip install -r requirements.txt

COPY debarta_training/. /opt/ml/debarta_training/
COPY dpr_training/. /opt/ml/dpr_training/
COPY lambda_function.py /opt/ml/lambda_function.py

# COPY new.sh /opt/ml/new.sh
