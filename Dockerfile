FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /tmp
COPY requirements.txt requirements.txt

RUN apt update && \
    apt install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --ignore-installed -r requirements.txt --no-cache-dir

RUN conda clean -a -y -f

WORKDIR /workspace
