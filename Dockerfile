FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /opt/ml

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends zip -y \
  && apt-get install -y --no-install-recommends python3.10 -y \
  && apt-get install -y --no-install-recommends python3-pip -y \
  && apt-get install -y --no-install-recommends nvidia-utils-550-server -y \
  && apt-get clean

COPY requirements.txt /opt/ml/requirements.txt

RUN pip install -r requirements.txt

COPY debarta_training/. /opt/ml/debarta_training/
COPY dpr_training/. /opt/ml/dpr_training/
COPY lambda_function.py /opt/ml/lambda_function.py
