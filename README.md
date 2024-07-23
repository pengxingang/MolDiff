# MolDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation (ICML 2023)
> This is **A Tailored Diffusion Framework for Generating 3D Drug-Like Molecules**, with sampling success rate of **>99\%**, almost threefold increase compared to the previous diffusion model. 

> This can serve as a better backbone for other applications of 3D molecule diffusion models such as pocket-based generation and linker generation.

More information can be found in our [paper](https://proceedings.mlr.press/v202/peng23b.html).


> **Update Jul 23, 2024**  
> Add the trained MolDiff checkpoint on the QM9 dataset and the corresponding sampling configuration file. See the `ckpt` directory.

## Installation
### Dependency
The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.8.13
PyTorch | 1.10.1
CUDA | 11.3.1
PyTorch Geometric | 2.0.4
RDKit | 2022.03.2


### Install via conda yaml file (cuda 11.3)
```bash
conda env create -f env.yaml
conda activate MolDiff
```

### Install manually

``` bash
conda create -n MolDiff python=3.8 # optinal, create a new environment
conda activate MolDiff

# Install PyTorch (for cuda 11.3)
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pyg -c pyg

# Install other tools
conda install -c conda-forge rdkit
conda install pyyaml easydict python-lmdb -c conda-forge

# Install tensorboard only for training
conda install tensorboard -c conda-forge
```


## Dataset

### Processed data
You can download the processed data `geom_drug.tar.gz` (~2GB) from [here](https://drive.google.com/drive/folders/1WkYIv471SjVwQe6_FfDxFOPC7dSnPY9c?usp=sharing) and unzip them (`tar -zxvf`)
 in the `data` folder as:
``` bash
data
├── geom_drug
│   ├── mol_summary.csv
│   ├── split_by_molid.pt
│   ├── processed.lmdb
│   └── processed_molid2idx.pt
```

### From sdf files
If you want to process the data from the sdf files, you have to further download the `sdf.tar.gz` (~4GB) from [here](https://drive.google.com/drive/folders/1WkYIv471SjVwQe6_FfDxFOPC7dSnPY9c?usp=sharing). Unzip it (~28GB after unzipping) in the `data/geom_drug` folder and remove the `processed.lmdb` and `processed_molid2idx.pt`:
``` bash
data
├── geom_drug
│   ├── mol_summary.csv
│   ├── split_by_molid.pt
│   └── sdf
│       ├── 0.sdf
│       ├── 1.sdf
│       ├── ...
```
Then by running any sampling, training or evaluation script, the data will be processed automatically.


## Sample

We provide the sampling config file `sample_MolDiff.yml` in `configs/sample` folder. We also provide a simplified version of MolDiff `sample_MolDiff_simple.yml` that does not use the bond guidance for sampling and uses the model trained without the new bond noise schedule proposed in the paper. 

To sample molecules using pretrained models, please first download the pretrained model weights from [here](https://drive.google.com/drive/folders/1zTrjVehEGTP7sN3DB5jaaUuMJ6Ah0-ps?usp=sharing) and put them in the `./ckpt` folder. There are three model weight files: 
- `MolDiff.pt`: the pretrained complete MolDiff model.
- `MolDiff_simple.pt`: the pretrained simplified MolDiff model that was trained without using the new bond noise schedule.
- `bond_predictor.pt`: the pretrained bond predictor that is used for bond guidance during sampling.

After preparing the pretrained weights (either the downloaded files or trained by yourself) and setting the correct model weight paths in the config file, you can run the following command to sample molecules:
```python
python scripts/sample_drug3d.py --outdir <output_directory> --config <path_to_config_file> --device <device_id> --batch_size <batch_size>
```
The parameters are:
- `outdir`: the root directory to save the sampled molecules.
- `config`: the path to the config file.
- `device`: the device to run the sampling.
- `batch_size`: the batch size for sampling. If set to 0 (default), it will use the batch size specified in the config file.

An example command is:
```python
python scripts/sample_drug3d.py --outdir ./outputs --config ./configs/sample/sample_MolDiff.yml
```
After sampling, there will be two directories in the `outdir` folder that contains the meta data and the sdf files of the sampling, respectively.

## Evaluate

To evaluate the generated molecules, run the following command:
```python
python scripts/evaluate_all.py --result_root <result_root> --exp_name <exp_name> --from_where generated
```
The parameters are:
- `result_root`: the parent directory of the directory of the sampled molecules (i.e, the same as the `outdir` parameter when running `sample_drug3d.py`).
- `exp_name`: the name (or prefix) of the directory of the molecules (excluding the suffix `_SDF`).
- `from_where`: be one of `generated` of `dataset`.

An example command to calculate metrics for the sampled molecules is:
```python
python scripts/evaluate_all.py --result_root ./outputs --exp_name sample_MolDiff_20230101_000000 --from_where generated
```

You also need to calculate some metrics for the test dataset to calculate the Jensen-Shannon divergence (JSD) between the generated molecules and the test dataset. To do so, run the following command:
```bash
python scripts/evaluate_all.py --exp_name test --from_where dataset
```

Then you can use the interactive notebook `script/analyze_generated.ipynb` to analyze all the metrics defined in the paper. But make sure to set the directory of the sampled molecules and the test dataset correctly in the notebook.

## Train

We also provide two versions of training config files: the complete MolDiff `train_MolDiff.yml` and the simplified one `train_MolDiff_simple.yml` (not use the new bond noise schedule). To train the model from scratch, run the following command:

```python
python scripts/train_drug3d.py --config <path_to_config_file> --device <device_id> --logdir <log_directory>
```
For example, to train the complete MolDiff model, run:
```python
python scripts/train_drug3d.py --config ./configs/train/train_MolDiff.yml --device cuda:0 --logdir ./logs
```

To use the bond predictor for guidance during sampling, you also need to train a bond predictor:
```python
python scripts/train_bond.py --config ./configs/train/train_bondpred.yml --device cuda:1 --logdir ./logs
```

## Citation
```bibtex
@InProceedings{pmlr-v202-peng23b,
  title =   {{M}ol{D}iff: Addressing the Atom-Bond Inconsistency Problem in 3{D} Molecule Diffusion Generation},
  author =       {Peng, Xingang and Guan, Jiaqi and Liu, Qiang and Ma, Jianzhu},
  booktitle =   {Proceedings of the 40th International Conference on Machine Learning},
  pages =   {27611--27629},
  year =   {2023},
  editor =   {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume =   {202},
  series =   {Proceedings of Machine Learning Research},
  month =   {23--29 Jul},
  publisher =    {PMLR},
  pdf =   {https://proceedings.mlr.press/v202/peng23b/peng23b.pdf},
  url =   {https://proceedings.mlr.press/v202/peng23b.html},
}
```

## Contact
If you have any question, feel free to contact me at xingang.peng@gmail.com


