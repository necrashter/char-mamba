# char-mamba

This repository contains a simple script for **Mamba-based Character-level Language Modeling**.
It can be considered the Mamba version of [char-rnn](https://github.com/karpathy/char-rnn).
Due to its simplicity, this script can serve as a **template for training Mamba models from scratch**, applicable to a wide array of sequence-to-sequence problems.

## Requirements

- CUDA
- [PyTorch](https://pytorch.org/get-started/locally/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/installation)
- [Mamba](https://github.com/state-spaces/mamba)

## Usage

`main.py` supports two subcommands: `train` and `generate`.

### Train

To get started, use the following command to train a simple model:
```bash
python main.py train --cut-dataset=100
```
This command will train Mamba on the first `100 * 256` characters of the Tiny Shakespeare dataset (downloading it if necessary) for 10 epochs, save the model, and produce a sample generation. It takes about 10 seconds on GTX 1650, and the resulting model is able to generate legitimate English words.

Once you make sure that it's working, you can train on the whole dataset by removing `--cut-dataset=100` argument. For more command line arguments, see the end of `main.py`.

The training code is based on [mamba-dive's fine-tuning script](https://github.com/Oxen-AI/mamba-dive/blob/main/train_mamba_with_context.py), which in turn is based on [mamba-chat](https://github.com/havenhq/mamba-chat).

### Generate

After training the model, you can use the `generate` subcommand to load the saved model and generate text:
```bash
python main.py generate
# Generate with a prompt:
python main.py generate --prompt=First
# Generate batched:
python main.py generate --batch=4
```

The generation code is based on [this script](https://github.com/state-spaces/mamba/blob/main/benchmarks/benchmark_generation_mamba_simple.py) and supports most of the same arguments.
