# SpikeHunter: A Deep Learning Tool for Identifying Phage Tailspike Proteins
SpikeHunter utilizes a deep learning approach to identify phage tailspike proteins. It achieves this by using a protein language model, ESM-2 transformer, which can detect patterns in protein sequences. The PyTorch framework enables efficient processing of large datasets, making SpikeHunter a valuable tool for researchers studying phage-host interactions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [License](#license)
- [Citation](#citation)

## Installation
To install SpikeHunter, follow these steps:

1. Install the required dependencies:
```
conda env create --file environment.yml
# or if you have mamba installed
# mamba env create --file environment.yml
```

2. Clone the repository:
```
git clone https://github.com/nlm-irp-jianglab/SpikeHunter.git
```

3. Change into the directory:
```
cd SpikeHunter
```
4. Download essential models
```
# get esm2 model
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

# please go to https://figshare.com/articles/online_resource/SpikeHunter_trained_model_pth_file/23577051 to download model_best.pth and copy it to trained_model/ folder
```

## Usage
```
usage: SpikeHunter.py [-h] [-c CONFIG] [-r RESUME] [-d DEVICE]

SpikeHunter

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
```

Simply run SpikeHunter using the following command:
```
# The input and output files are in folder `test/`
python SpikeHunter.py -c config.yaml -r trained_model/model_best.pth
```

## Features

- Identifies tailspike proteins in phage sequences using advanced algorithms.
- Supports analysis of multiple phage genomes in a single run.
- Exports results in a TSV format for easy downstream analysis.

## Dependencies

SpikeHunter requires the following Python libraries:

- Pytorch
- Pandas
- Numpy

## License

SpikeHunter is released under the [MIT License](./LICENSE).

## Citation

If you use SpikeHunter in your research, please cite it as follows:
Yang Y, Dufault-Thompson K, Yan W, Cai T, Xie L, Jiang X. Deciphering Phage-Host Specificity Based on the Association of Phage Depolymerases and Bacterial Surface Glycan with Deep Learning. bioRxiv. 2023:2023-06.

## Credits

This project uses code adapted from the [pytorch-template](https://github.com/victoresque/pytorch-template) repository by Victor Huang. We thank Victor Huang for providing this useful code as a starting point for our project.
