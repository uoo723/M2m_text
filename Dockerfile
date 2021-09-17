FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /tmp

RUN apt update && \
    apt install -y --no-install-recommends build-essential git zsh curl vim less && \
    rm -rf /var/lib/apt/lists/* && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    chsh -s $(which zsh) && \
    conda init zsh

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
    ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/djui/alias-tips.git \
    ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/alias-tips && \
    perl -pi -e 's/plugins\=\(git\)/plugins\=\(git zsh-syntax-highlighting alias-tips\)/g' ~/.zshrc

COPY requirements.txt requirements.txt

RUN pip install --ignore-installed -r requirements.txt --no-cache-dir

RUN conda clean -a -y -f

WORKDIR /workspace
