# SupER (Repository Preview)

This repository is a preview codebase for our CPAL 2026 paper [Superclass-Guided Representation Disentanglement for Spurious Correlation Mitigation](https://arxiv.org/pdf/2508.08570). A fully polished and final release of the code is coming soon.

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Datasets

This repo supports the following datasets:

- Waterbirds
- MetaShift
- Spawrious
- SpuCoDogs

Please download each dataset from its original source and place it under the corresponding folder in `./data/`:

```text
./data/waterbirds/
./data/metashift/
./data/spawrious/
./data/spucodogs/
```

For convenience, you may also download the packaged datasets from [here](https://drive.google.com/drive/folders/1avkaSP9jMH5sj8RySXFlDvkpjIWD-HAp?usp=sharing).

---

## Run

Run commands from the repo root:

### Waterbirds
```bash
python src/Waterbirds.py --dataset waterbirds_0.95
python src/Waterbirds.py --dataset waterbirds_1.0
```

### Spawrious
```bash
python src/Spawrious.py --dataset o2o_easy
python src/Spawrious.py --dataset o2o_medium
python src/Spawrious.py --dataset o2o_hard

python src/Spawrious.py --dataset m2m_easy
python src/Spawrious.py --dataset m2m_medium
python src/Spawrious.py --dataset m2m_hard
```

### MetaShift
```bash
python src/Metashift.py --dataset metashift_a
python src/Metashift.py --dataset metashift_b
python src/Metashift.py --dataset metashift_c
python src/Metashift.py --dataset metashift_d
```

### SpuCoDogs
```bash
python src/Spucodogs.py
```

---

## Outputs

By default, each script writes:

- a `*.log` training log  
- a `*.pth` checkpoint

These files will appear under:

```text
./outputs/
```
