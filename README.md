# M2m_text

## Prerequisites

* python==3.7.9
* torch==1.7.1
* numpy==1.19.2
* tqdm==4.54.1
* click==7.1.2
* logzero==1.6.3
* scikit-learn==0.24.1
* ruamel.yaml==0.16.12
* gdown==3.12.2
* pandas==1.1.5
* nltk==3.5
* gensim==3.8.3
* scipy==1.5.2
* transformers==4.2.1
* mlflow==1.15.0

```bash
$ pip install -r requirements.txt
```

## Docker

If you prefer to use docker, follow below instructions.

### Prerequisites

* cuda >= 11.0
* [Docker](https://www.docker.com) >= 19.03.5
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

#### Build Image

```bash
$ docker build -t [image_name]
```

Or use our pre-built image [realappsdev/m2m-text](https://hub.docker.com/repository/docker/realappsdev/m2m-text)

#### Run docker cotainer

```bash
$ docker run --rm -t -d --init --name [container_name] --gpus all -v $PWD:/workspace -w /workspace --ipc=host --net=host [image_name]
$ docker exec -it [cotainer_name] zsh  # Enter shell
```

## Run

```bash
$ ./scripts/run_model.sh
```

## MLflow Integration

You can track experiments with mlflow.

### Run MLflow UI

```bash
$ ./scripts/mlflow_ui.sh  # Enter http://[your-ip]:5000
```
