# PTM-AGT: A Post-Translational Modification-Guided Adaptive Graph Transformer Network for Cancer Subtype Classification

PTM-AGT classifies cancer subtypes from RPPA protein expression profiles. Features shared with a
curated PTM–disease association matrix are reweighted before a patient similarity graph is encoded
by a graph-constrained Transformer, and the encoder depths are fused by a learnable layer attention.

## Datasets

Three TCGA RPPA cohorts (GDAC Firehose, robustly scaled) together with the PTM–disease association
matrix of PhosphoSitePlus and PTMD v1.0:

| Cohort | Subtypes | Samples | Proteins |
| --- | --- | --- | --- |
| COADREAD | COAD / READ | 488 | 196 |
| KIPAN | KC / KRCC / KRPC | 752 | 178 |
| THCA | Classical / other | 221 | 175 |

```text
dataset/
├── association/
│   ├── Association_degree_score_matrix.xlsx   PTM–disease association scores
│   └── protein_numbers.xlsx                   protein identifiers
├── COADREAD/
│   ├── samples_protein.xlsx
│   ├── labels.xlsx
│   └── feature_name.xlsx
├── KIPAN/
└── THCA/
```

## Project Structure

```text
PTMAGT/
├── model/              # PTM-AGT encoder and classifier
├── preprocess/         # data loading, prior weighting, graph construction
├── utils/              # training loop, metrics, seeding
├── dataset/            # the three cohorts and the association matrix
├── main.py             # five-fold cross-validation
├── test.py             # evaluation of a saved checkpoint
├── aggregate.py        # multi-seed summary
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
git clone https://github.com/wenhaoyang-nyist/PTM-AGT.git
cd PTM-AGT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9 and PyTorch 2.0.0+cu118; `--device cpu` is also supported.

## Training

Run the five-fold cross-validation of the paper on all three cohorts:

```bash
python main.py --dataset all --seed 42
```

A single cohort, and evaluation of a saved fold checkpoint:

```bash
python main.py --dataset THCA --seed 42 --save-checkpoints
python test.py --dataset THCA --seed 42 --fold 1 \
    --checkpoint results/THCA/checkpoints/seed42_fold1.pt
```

Pool the results of several seeds:

```bash
python aggregate.py
```

Fold scores are written to `results/<cohort>/seed<seed>_folds.csv`. Use `--help` for the remaining
options (epochs, learning rates, prior strength, model width).

## Notes

- `samples_protein.xlsx` and `labels.xlsx` carry no header row, so the default `header=0` consumes
  their first row; this reproduces the reported numbers (488 / 752 / 221 samples). Pass
  `--keep-first-row` to use every stored row.
- Runs are deterministic for a fixed seed. The paper averages five seeds (42, 100, 123, 456, 789)
  over five folds.

## Data Availability

The processed datasets — the three RPPA cohorts and the PTM–disease association matrix — are
archived on Zenodo under a citable DOI:

> **PTM-AGT dataset v1.0.0** — <https://doi.org/10.5281/zenodo.24301397>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24301397.svg)](https://doi.org/10.5281/zenodo.24301397)

## Citation

> Liu W, Yang W, Li K, Guo L, Zhan Y, Huang H, Zhang R. PTM-AGT: a post-translational
> modification-guided adaptive graph Transformer network for cancer subtype classification.
> *(journal and volume to be added)*

If you reuse the data, please cite the dataset DOI above in addition to the paper.

Please contact Wenhao Yang (wenhaoyang@nyist.edu.cn) for questions or corrections.
