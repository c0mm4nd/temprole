# CryptoFIRM

## Ready

install the pytorch and pytorch-geometric version according to your system from [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

```
git clone <RPPO_URL> && cd CryptoFIRM
python -m pip install networkx tqdm numpy
```

if you found any other missing packages, please install them using `python -m pip install <package_name>`

## Datasets

- [Russia Ukraine](./russia_ukraine_dataset_construction/README.md)
- [Palestine Israel](./israel_palestine_dataset_construction/README.md)

## Run

```bash
python run_cryptofirm_with_step.py --dataset palestine_israel_pruned --train-batch-size 128
python run_cryptofirm_with_step.py --dataset russia_ukraine_pruned --train-batch-size 64
```
