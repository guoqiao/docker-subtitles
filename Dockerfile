# FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# git is needed to install pkg via repo
RUN apt-get update -qq \
    && apt-get install -y -qq \
        git \
        time \
        tree \
    && apt-get clean

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt