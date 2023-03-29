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
To be added
