FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /tmp

RUN apt update && \
    apt install -y --no-install-recommends build-essential git zsh curl && \
    rm -rf /var/lib/apt/lists/* && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    chsh -s $(which zsh) && \
    conda init zsh

COPY requirements.txt requirements.txt

RUN pip install --ignore-installed -r requirements.txt --no-cache-dir

RUN conda clean -a -y -f

WORKDIR /workspace
