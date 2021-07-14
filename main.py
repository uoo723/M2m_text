"""
Created on 2020/12/31
@author Sangwoo Han
"""
import copy
import os
import random
import time
import warnings
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

import click
import logzero
import mlflow
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logzero import logger
from ruamel.yaml import YAML
from scipy.sparse import csr_matrix, csc_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from m2m_text.datasets import (
    RCV1,
    AmazonCat,
    AmazonCat13K,
    DrugReview,
    DrugReviewSmall,
    DrugReviewSmallv2,
    EURLex,
    EURLex4K,
    Wiki10,
    Wiki10_31K,
)
from m2m_text.datasets._base import Dataset
from m2m_text.metrics import get_inv_propensity
from m2m_text.networks import (
    AttentionRGCN,
    AttentionRNN,
    AttentionRNNEncoder,
    SBert,
    CornetAttentionRNN,
    CornetAttentionRNNv2,
    EaseAttentionRNN,
    FCNet,
    LabelGCNAttentionRNN,
    LabelGCNAttentionRNNv2,
    LabelGCNAttentionRNNv3,
    LabelGCNAttentionRNNv4,
    LaRoberta,
    LaRobertaV2,
    RobertaForSeqClassification,
)
from m2m_text.optimizers import DenseSparseAdam
from m2m_text.train import evaluate, train
from m2m_text.utils.data import (
    get_dense_label_features,
    get_le,
    get_mlb,
    get_n_samples_per_class,
    get_oversampled_data,
)
from m2m_text.utils.graph import get_adj, get_ease_weight, get_random_adj
from m2m_text.utils.mlflow import log_ckpt, log_config, log_logfile, log_metric, log_tag
from m2m_text.utils.model import load_checkpoint

from sentence_transformers import SentenceTransformer, models
from m2m_text.cluster import get_clusters
from m2m_text.utils.data import get_sparse_features
from sklearn.preprocessing import MultiLabelBinarizer, normalize


MODEL_CLS = {
    "AttentionRNN": AttentionRNN,
    "AttentionRNNEncoder": AttentionRNNEncoder,
    "SBert": SBert,
    "AttentionRGCN": AttentionRGCN,
    "CornetAttentionRNN": CornetAttentionRNN,
    "CornetAttentionRNNv2": CornetAttentionRNNv2,
    "FCNet": FCNet,
    "Roberta": RobertaForSeqClassification,
    "LaRoberta": LaRoberta,
    "LaRobertaV2": LaRobertaV2,
    "EaseAttentionRNN": EaseAttentionRNN,
    "LabelGCNAttentionRNN": LabelGCNAttentionRNN,
    "LabelGCNAttentionRNNv2": LabelGCNAttentionRNNv2,
    "LabelGCNAttentionRNNv3": LabelGCNAttentionRNNv3,
    "LabelGCNAttentionRNNv4": LabelGCNAttentionRNNv4,
    # "LabelGCNAttentionRNNv5": LabelGCNAttentionRNNv5,
}

TRANSFORMER_MODELS = ["Roberta", "LaRoberta", "LaRobertaV2"]

DATASET_CLS = {
    "DrugReview": DrugReview,
    "RCV1": RCV1,
    "DrugReviewSmall": DrugReviewSmall,
    "DrugReviewSmallv2": DrugReviewSmallv2,
    "EURLex": EURLex,
    "EURLex4K": EURLex4K,
    "AmazonCat": AmazonCat,
    "Wiki10": Wiki10,
    "AmazonCat13K": AmazonCat13K,
    "Wiki10_31K": Wiki10_31K,
}

MULTI_LABEL_DATASETS = ["EURLex", "EURLex4K", "AmazonCat", "Wiki10", "Wiki10_31K"]

GCN_MODELS = [
    "LabelGCNAttentionRNN",
    "LabelGCNAttentionRNNv2",
    "LabelGCNAttentionRNNv3",
    "LabelGCNAttentionRNNv4",
    # "LabelGCNAttentionRNNv5",
]

#decorator
def log_elapsed_time(func):
    '''함수 실행 전후 시간차를 구해 함수를 실행하는 데 걸린 시간 기록'''
    @wraps(func)
    #decorator debugging을 쉽게 하기 위해 @wraps 이용.
    #데코레이터를 작성하는 데 이용하는 데코레이터.
    #내부함수(func)에 있는 중요 메타데이터를 모두 외부함수(log_elapsed_time)로 복사해줌. 
    #파이썬 함수의 여러 표준 속성(__doc__, __name__, __help__)들을 보호.  
    def wrapper(*args, **kwargs):
        #*args: 여러 개의 인자를 튜플로 저장 **kwargs: 여러개의 key=value를 dictionary로 저장 
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.info(f"elapsed time: {end - start:.2f}s, {timedelta(seconds=elapsed)}")

        return ret

    return wrapper


def set_mlflow_status(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt as e: # cntrl + c로 종료시 exception
            run = mlflow.active_run()
            if run is not None:
                mlflow.end_run("KILLED")
            raise e
        except Exception as e:
            run = mlflow.active_run()
            if run is not None:
                mlflow.end_run("FAILED")
            raise e

    return wrapper


def set_logger(log_path: str):
    #exist_ok=True:이미 해당 디렉토리가 존재하는 경우 무시하고 넘어감, 없을 경우 새로 생성. 
    #os.path.dirname:해당 파일 또는 폴더가 속한 상위 폴더의 경로 표시 
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # start writing into a logfile 
    logzero.logfile(log_path)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model(
    model_name: str,
    num_labels: int,
    model_cnf: dict,
    data_cnf: dict,
    dataset: Optional[Dataset] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
):
    model_cnf = copy.deepcopy(model_cnf)
    #dict: mutable한 type
    #shallow copy,, copy.copy: 변수간 대입..ex) a = b 
        #*mutable한 객체(list, set 등)의 경우: copy 후 b 재할당 시 a, b가 같이 바뀜 (메모리 주소 같음)
        #*immutable한 객체(tuple, str 등)의 경우: copy 후 b 재할당 시 b만 바뀜 (메모리 주소 다름)

    #deep copy 
        #mutable한 객체를 immutable한 객체처럼 메모리 주소를 다르게 할당해줌. 

    model_cnf["model"]["max_length"] = data_cnf["dataset"]["maxlen"]

    if model_name in TRANSFORMER_MODELS:
        pretrained_model_name = model_cnf["model"].pop("pretrained")
        network = MODEL_CLS[model_name].from_pretrained(
            pretrained_model_name, num_labels=num_labels, **model_cnf["model"]
        )
    else: #transformer 기반 모델이 아니면 {data}.npy 임베딩으로 초기화. 
        model_cnf["model"]["emb_init"] = data_cnf["model"]["emb_init"]

        if model_name == "EaseAttentionRNN":
            model_cnf["model"]["dataset"] = dataset
            model_cnf["model"]["device"] = device

        if model_name in GCN_MODELS:
            random_adj_init_sp = model_cnf.pop("random_adj_init_sp", None)
            if random_adj_init_sp is not None:
                adj = get_random_adj(num_labels, random_adj_init_sp)
            else:
                lamda = model_cnf["model"].pop("lamda")
                top_adj = model_cnf["model"].pop("top_adj")
                use_b_weights = model_cnf["model"].pop("use_b_weights", False)
                laplacian_norm = model_cnf["model"].pop("laplacian_norm", True)
                b = get_ease_weight(dataset, lamda)
                adj = get_adj(b, top_adj, use_b_weights, laplacian_norm)

            if model_cnf["model"].pop("label_emb_init", False):
                if verbose:
                    logger.info("Get label embeddings")
                model_cnf["model"]["label_emb_init"] = get_dense_label_features(
                    dataset.emb_init_path, dataset.x, dataset.y
                )

            if verbose:
                sparsity = np.count_nonzero(adj) / adj.shape[0] ** 2
                logger.info(f"Sparsity of label adj: {1 - sparsity:.8f}")
            model_cnf["model"]["gcn_init_adj"] = torch.from_numpy(adj).float()

        network = MODEL_CLS[model_name](num_labels=num_labels, **model_cnf["model"])

        if model_name == "SBert":
            sbert_pretrained = SentenceTransformer("stsb-roberta-base-v2")
            network.model[0].load_state_dict(list(sbert_pretrained.modules())[0][0].state_dict())

    return network


def get_optimizer(model_name: str, network: nn.Module, lr: float, decay: float):
    if model_name in TRANSFORMER_MODELS:
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [
                    p
                    for n, p in network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": decay,
            },
            {
                "params": [
                    p
                    for n, p in network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = DenseSparseAdam(param_groups, lr=lr)
    else:
        optimizer = DenseSparseAdam(network.parameters(), lr=lr, weight_decay=decay)

    return optimizer

def get_dataset(
    model_name: str,
    dataset_name: str,
    model_cnf: dict,
    data_cnf: dict,
    do_clustering: bool = False,
    ):
    if do_clustering:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_dataset = DATASET_CLS[dataset_name](
                **data_cnf["dataset"], **model_cnf.get("dataset", {})
            )
            valid_dataset = copy.deepcopy(train_dataset)
            test_dataset = DATASET_CLS[dataset_name](
                train=False, **data_cnf["dataset"], **model_cnf.get("dataset", {})
            )
        
        train_texts = np.array(
            [
                " ".join(texts)
                for texts in np.load(train_dataset.tokenized_path, allow_pickle=True)
            ],
            dtype=np.object,
        )
        test_texts = np.array(
            [
                " ".join(texts)
                for texts in np.load(test_dataset.tokenized_path, allow_pickle=True)
            ],
            dtype=np.object,
        )

        train_mask = torch.cat(
            [
                torch.zeros(train_dataset.x.shape[0], dtype=torch.bool).bernoulli(
                    1 - data_cnf["valid_size"]
                ),
            ]
        ) #tensor([False,  True,  True,  ...,  True,  True,  True])    

        if model_name == "SBert":
            word_embedding_model = models.Transformer(
                model_cnf["model"]["model_name"], max_seq_length=data_cnf["dataset"]["maxlen"]
            )
            embedding_size = word_embedding_model.get_word_embedding_dimension()

            pooling_model = models.Pooling(
                embedding_size,
                pooling_mode=None,
            )
            sbert_pretrained = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            
            train_tokenized_texts = sbert_pretrained.tokenize(train_texts[train_mask])
            #"input_ids" length : 150
            # for i in range(5):
            #     print(len(train_tokenized_texts["input_ids"][i]), len(train_tokenized_texts["attention_mask"][i]) )
            valid_tokenized_texts = sbert_pretrained.tokenize(train_texts[~train_mask])
            test_tokenized_texts = sbert_pretrained.tokenize(test_texts)
            
            train_dataset.x = train_tokenized_texts
            train_dataset._split_indices = train_mask

            valid_dataset.x = valid_tokenized_texts
            valid_dataset._split_indices = ~train_mask

            test_dataset.x = test_tokenized_texts
        else:
            train_dataset.x = train_dataset.x[train_mask]
            train_dataset._split_indices = train_mask

            valid_dataset.x = valid_dataset.x[~train_mask]
            valid_dataset._split_indices = ~train_mask

        train_cluster_y, test_cluster_y, train_cluster_raw_y, test_cluster_raw_y, num_clusters, cluster_mlb = get_cluster(dataset_name, train_texts, train_dataset, test_dataset)

        train_dataset.y = train_cluster_y[train_mask]
        train_dataset.raw_y = train_cluster_raw_y[train_mask]

        valid_dataset.y = train_cluster_y[~train_mask]
        valid_dataset.raw_y = train_cluster_raw_y[~train_mask]

        test_dataset.y = test_cluster_y
        test_dataset.raw_y = test_cluster_raw_y

        num_labels = num_clusters
        mlb = cluster_mlb

    else:
        train_dataset, valid_dataset = DATASET_CLS[dataset_name].splits(
            test_size=data_cnf.get("valid_size", 200),
            **data_cnf["dataset"],
            **model_cnf.get("dataset", {}),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_dataset = DATASET_CLS[dataset_name](
                train=False,
                **data_cnf["dataset"],
                **model_cnf.get("dataset", {}),
            )
        
        mlb = get_mlb(train_dataset.le_path)
        le = get_le(train_dataset.le_path)
        num_labels = len(le.classes_)

    logger.info(f"# of train dataset: {len(train_dataset):,}")
    logger.info(f"# of valid dataset: {len(valid_dataset):,}")
    logger.info(f"# of test dataset: {len(test_dataset):,}")
    logger.info(f"# of classes: {num_labels:,}")

    return train_dataset, valid_dataset, test_dataset, num_labels, mlb

def get_cluster(dataset_name, train_tokenized_texts, train_dataset, test_dataset): 
    """
    data 하나에 대응하는 label들이 속한 cluster ids
    모든 train, test data에 대해서 csr_matrix로 저장. 
    """
    dataset_path = os.path.abspath(os.path.join("./data", dataset_name))
    filepath = os.path.join(dataset_path, "train_sparse.npz")

    sparse_x = get_sparse_features(filepath, train_tokenized_texts, force=False) #shape (data 개수, corpus 개수)
    sparse_y = train_dataset.y
    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x)) # (data 개수, label 개수).T x (data 개수, corpus 개수) = (label 개수, corpus 개수)
    clusters = get_clusters(labels_f, levels=[11], verbose=True)

    cluster = clusters[0] #index = cluster, value = label_ids 

    num_clusters = len(cluster)
    #print(f"# cluster:{num_clusters}") #512

    label_to_cluster = np.zeros(labels_f.shape[0], dtype=np.int64)

    for i, label_ids in enumerate(cluster):
        """
        labels                                          cluster
        │[1698  855 1813 2190   29 2473 2477  450],     171
        │[ 293 3590 1329  969 1987  878 2242],          172
        │[2363 2403  226 1812 2547  252 2860 3349],     173
        │[ 424 2043 3715 2637 3006 1196 2819],          174
        │[3411 1877  163   81 2565 2417  246  989],     175
        │[3177  579  584 2331 2328  221 3069],          176
        │[3658 3592 3659 3047 3729 2248 2251],          177
        │[ 107  103 2601  526 1538  973 2410],          178
        │[3504 3655 2412 1833 2411  125 3416 3430],     179
      
        """
        #if i>170 and i<180: print(f"{label_ids}, {i}")
        label_to_cluster[label_ids] = i

    ########################for train_dataset##############################
    
    cluster_data = []
    cluster_rows = []
    cluster_cols = []
    train_cluster_y = []

    for i, y in enumerate(sparse_y):
        #c_ids: data 하나에 대응하는 label들이 속한 cluster ids
        c_ids = np.unique(label_to_cluster[y.indices]) #겹치는 값 빼고 list로 반환해줌  
        #if i>400 and i<415: print(f"y.indices: {y.indices}, label_to_cluster:{label_to_cluster[y.indices]}, c_ids:{c_ids}")

        cluster_data.extend([1.0 for _ in range(c_ids.shape[0])])
        cluster_rows.extend([i for _ in range(c_ids.shape[0])])
        cluster_cols.extend(c_ids.tolist())

        train_cluster_y.append(c_ids)

    train_cluster_y = np.array(train_cluster_y, dtype=np.object)
    train_cluster_sparse_y = csr_matrix(
        (cluster_data, (cluster_rows, cluster_cols)), shape=(sparse_y.shape[0], num_clusters)
    )
    ########################for test_dataset##############################

    cluster_data = []
    cluster_rows = []
    cluster_cols = []
    test_cluster_y = []

    for i, y in enumerate(test_dataset.y):
        c_ids = np.unique(label_to_cluster[y.indices])

        cluster_data.extend([1.0 for _ in range(c_ids.shape[0])])
        cluster_rows.extend([i for _ in range(c_ids.shape[0])])
        cluster_cols.extend(c_ids.tolist())

        test_cluster_y.append(c_ids)

    test_cluster_y = np.array(test_cluster_y, dtype=np.object)
    test_cluster_sparse_y = csr_matrix(
        (cluster_data, (cluster_rows, cluster_cols)),
        shape=(test_dataset.y.shape[0], num_clusters),
    )

    cluster_mlb = MultiLabelBinarizer(
        classes=np.arange(num_clusters), sparse_output=True
        ).fit(train_cluster_y)
    
    return train_cluster_sparse_y, test_cluster_sparse_y, train_cluster_y, test_cluster_y, num_clusters, cluster_mlb

@click.command(context_settings={"show_default": True})
@click.option(
    "--mode",
    type=click.Choice(["train", "eval"]),
    default="train",
    help="train: train and eval are executed. eval: eval only",
)
@click.option("--test-run", is_flag=True, default=False, help="Test run mode for debug")
@click.option("--log-dir", type=click.Path(), default="./logs", help="Log dir")
@click.option("--seed", type=click.INT, default=0, help="Seed for reproducibility")
@click.option(
    "--model-cnf", type=click.Path(exists=True), help="Model config file path"
)
@click.option("--data-cnf", type=click.Path(exists=True), help="Data config file path")
@click.option(
    "--run-script", type=click.Path(exists=True), help="Run script file path to log"
)
@click.option(
    "--ckpt-root-path",
    type=click.Path(),
    default="./checkpoint",
    help="Checkpoint root path",
)
@click.option("--ckpt-name", type=click.STRING, help="Checkpoint name")
@click.option("--net-t", type=click.STRING, help="Checkpoint path of training network")
@click.option(
    "--num-workers", type=click.INT, default=4, help="Number of workers for data loader"
)
@click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda")
@click.option(
    "--train-batch-size", type=click.INT, default=128, help="Batch size for training"
)
@click.option(
    "--test-batch-size", type=click.INT, default=256, help="Batch size for test"
)
@click.option("--decay", type=click.FLOAT, default=2e-4, help="Weight decay")
@click.option("--lr", type=click.FLOAT, default=0.1, help="learning rate")
@click.option(
    "--eta-min",
    type=click.FLOAT,
    default=1e-4,
    help="Minimum learning rate for cosine annealing scheduler",
)
@click.option("--no-scheduler", is_flag=True, default=False, help="Disable scheduler")
@click.option("--epoch", type=click.INT, default=200, help="Total number of epochs")
@click.option(
    "--swa-warmup", type=click.INT, default=10, help="Warmup for SWA. Disable: -1"
)
@click.option(
    "--gradient-max-norm",
    type=click.FLOAT,
    help="max norm for gradient clipping",
)
@click.option(
    "--eval-step",
    type=click.INT,
    default=100,
    help="Evaluation step during training",
)
@click.option(
    "--early",
    type=click.INT,
    default=50,
    help="Early stopping step",
)
@click.option(
    "--early-criterion",
    type=click.Choice(["p1", "p3", "p5", "n1", "n3", "n5", "psp1", "psp3", "psp5"]),
    default="n5",
    help="Early stopping criterion",
)
@click.option(
    "--do-clustering", is_flag=True, default=False, help="Do clustering") #is_flag이기 때문에 .sh에 값을 넣지 않아도 True로 인식함.
@log_elapsed_time
@set_mlflow_status
def main(
    mode,
    test_run,
    log_dir,
    seed,
    model_cnf,
    data_cnf,
    run_script,
    ckpt_root_path,# "./checkpoint"
    ckpt_name,# "baseline" etc.
    net_t,# Checkpoint path of training network
    num_workers,
    no_cuda,
    train_batch_size,
    test_batch_size,
    decay,
    lr,
    eta_min,
    no_scheduler,
    epoch,
    swa_warmup,
    gradient_max_norm,
    eval_step,
    early,
    early_criterion,
    do_clustering,
):
    yaml = YAML(typ="safe")

    model_cnf_path = model_cnf #hidden_size, num_layer, dropout 등이 저장되어 있음. 
    data_cnf_path = data_cnf #max_len, valid_size 등이 저장되어 있음. 

    model_cnf = yaml.load(Path(model_cnf))
    data_cnf = yaml.load(Path(data_cnf))

    os.makedirs(ckpt_root_path, exist_ok=True)

    model_name = model_cnf["name"]
    dataset_name = data_cnf["name"]

    prefix = "" if ckpt_name is None else f"{ckpt_name}_"
    ckpt_name = f"{prefix}{model_name}_{dataset_name}_{seed}.pt"
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)

    log_filename = os.path.splitext(ckpt_name)[0] + ".log"

    if not test_run:
        mlflow.start_run()
        log_config(data_cnf_path, model_cnf_path, run_script, test_run)
        log_tag(model_name, dataset_name, prefix, seed, test_run)
        set_logger(os.path.join(log_dir, log_filename))

    if seed is not None:
        logger.info(f"seed: {seed}")
        set_seed(seed)

    device = torch.device("cpu" if no_cuda else "cuda")
    num_gpus = torch.cuda.device_count()

    is_transformer = model_name in TRANSFORMER_MODELS
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    ################################## Prepare Dataset ###############################
    data_mlb = None

    logger.info(f"Dataset: {dataset_name}")

    train_dataset, valid_dataset, test_dataset, num_labels, data_mlb = get_dataset(model_name, dataset_name, model_cnf, data_cnf, do_clustering)

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=False if no_cuda else True,
        batch_size=train_batch_size,
    )

    valid_loader = DataLoader(
        valid_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False if no_cuda else True,
        batch_size=test_batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False if no_cuda else True,
        batch_size=test_batch_size,
    )
    ##################################################################################
    ################################# Prepare Model ##################################
    logger.info(f"Model: {model_name}")

    network = load_model(
        model_name, num_labels, model_cnf, data_cnf, train_dataset, device, verbose=True
    )

    network.to(device)

    if num_gpus > 1 and not no_cuda:
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs")
        network = nn.DataParallel(network)
    elif not no_cuda:
        logger.info("Single-GPU mode")
    else:
        logger.info("CPU mode")
    ##################################################################################
    ################################### Training #####################################
    if mode == "train":
        criteron = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        optimizer = get_optimizer(model_name, network, lr, decay)
        if not no_scheduler:
            #lr이 eta_min까지 떨어졌다 다시 초기 lr까지 올라온다. 
            #lr은 T_max*2 단위로 반복한다. 
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(3, epoch // 10), eta_min=eta_min
            )
        else:
            scheduler = None

        if net_t is not None:
            logger.info("Resume training")
            start_epoch, other_states = load_checkpoint(
                net_t, network, optimizer, scheduler, return_other_states=True
            )
            start_epoch += 1
        else:
            start_epoch = 0
            other_states = {}

        logger.info("Training")
        logger.info(os.path.splitext(os.path.basename(ckpt_path))[0])
        train(
            network,
            device,
            test_run, #Test run mode for debug
            start_epoch,
            epoch,
            optimizer,
            scheduler,
            criteron,
            swa_warmup,
            gradient_max_norm,
            train_loader,
            valid_loader,
            ckpt_path,
            data_mlb, 
            other_states=other_states,
            step=eval_step,
            early=early,
            early_criterion=early_criterion,
            is_transformer=is_transformer,
            multi_label=multi_label,
            **model_cnf.get("train", {}),
        )
    ##################################################################################

    ################################### Evaluation ###################################
    if mode == "eval":
        ckpt_path = net_t

    logger.info("Evaluation")

    load_checkpoint(ckpt_path, network, set_rng_state=False)

    if multi_label:
        inv_w = get_inv_propensity(sp.vstack([train_dataset.y, valid_dataset.y]))
        mlb = data_mlb
    else:
        inv_w = None
        mlb = None

    logger.info(os.path.splitext(os.path.basename(ckpt_path))[0])

    evaluate(
        network,
        test_loader,
        num_labels,
        device,
        is_transformer,
        multi_label,
        inv_w,
        mlb,
        test_run,
    )

    log_logfile(os.path.join(log_dir, log_filename), test_run)
    log_ckpt(ckpt_path, test_run)
    ##################################################################################


if __name__ == "__main__":
    main()
