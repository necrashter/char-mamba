import torch
import argparse
import json
import os
import pickle
from urllib.request import urlretrieve
from dataclasses import  asdict
from torch.utils.data import TensorDataset
from transformers import TrainingArguments, Trainer
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Adds the methods needed by Trainer
class MyMambaConfig(MambaConfig):
    def to_json_string(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)


class CharTokenizer:
    def __init__(self, chars):
        self.char_to_token = {char: i for i, char in enumerate(chars)}
        self.token_to_char = {i: char for char, i in self.char_to_token.items()}
        self.vocab_size = len(self.char_to_token)

    def encode(self, string):
        return [self.char_to_token[i] for i in string]

    def decode(self, tokens):
        return "".join([self.token_to_char[i] for i in tokens])


class MambaTrainer(Trainer):
    def __init__(self, **kwargs):
        super(MambaTrainer, self).__init__(**kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        lm_logits = model(inputs).logits
        labels = inputs.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        return self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

    def save_model(self, output_dir, _internal_call=None):
        if not output_dir:
            output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            f.write(self.model.config.to_json_string())

def train(args):
    # Download dataset if required
    if (data_path := args.data_path) is None:
        if not os.path.exists(data_path := "input.txt"):
            print("Dataset not found, downloading tinyshakespeare...")
            urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_path)

    # Load dataset
    with open(data_path, "r") as file:
        content = file.read()
    if args.cut_dataset is not None:
        content = content[:args.train_length*args.cut_dataset]
    tokenizer = CharTokenizer(set(content))
    dataset = TensorDataset(torch.stack([
        torch.LongTensor(tokenizer.encode(content[i:i+args.train_length]))
        for i in range(0, len(content)-args.train_length+1, args.train_length)
    ]))

    # Save tokenizer
    output_dir = args.model_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

    config = MyMambaConfig(
        d_model = args.d_model,
        n_layer = args.n_layer,
        vocab_size = tokenizer.vocab_size,
    )
    model = MambaLMHeadModel(config, device="cuda")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:_}")

    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=output_dir,
            logging_steps=50,
            save_steps=500,
        ),
        # TensorDataset yields a list of 1-tuples. Unpack the tuples and stack them.
        data_collator=lambda instances: torch.stack([i[0] for i in instances]),
    )
    trainer.train()
    trainer.save_model(args.model_path)

    print("Sample generation:")
    out = model.generate(
        input_ids=torch.randint(0, tokenizer.vocab_size, (1, 1), dtype=torch.long, device="cuda"),
        max_length=200,
    )
    print(tokenizer.decode(out[0].tolist()))


def generate(args):
    with open(os.path.join(args.model_path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer: CharTokenizer = pickle.load(f)
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config = MyMambaConfig(**json.load(f))
    model = MambaLMHeadModel(config, device="cuda")
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    model.eval()

    if args.prompt is None:
        input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    else:
        input_ids = torch.LongTensor(tokenizer.encode(args.prompt)).repeat(args.batch, 1).to(device="cuda")

    out = model.generate(
        input_ids=input_ids,
        max_length=input_ids.shape[1] + args.length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
    print("\n======\n".join([tokenizer.decode(seq) for seq in out.sequences.tolist()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model")

    subparsers = parser.add_subparsers(dest="subparser", required=True)

    train_parser = subparsers.add_parser("train")
    # Train config
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--optim", type=str, default="adamw_torch")
    train_parser.add_argument("--data-path", type=str)
    train_parser.add_argument("--num-epochs", type=int, default=10)
    train_parser.add_argument("--train-length", type=int, default=256, help="Sequence length of a single sample in training")
    train_parser.add_argument("--cut-dataset", type=int, help="Limit the number of samples in the dataset")
    # Model config
    train_parser.add_argument("--d_model", type=int, default=256)
    train_parser.add_argument("--n_layer", type=int, default=6)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--prompt", type=str, default=None)
    generate_parser.add_argument("--promptlen", type=int, default=1)
    generate_parser.add_argument("--length", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--topk", type=int, default=1)
    generate_parser.add_argument("--topp", type=float, default=1.0)
    generate_parser.add_argument("--minp", type=float, default=0.0)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()
    globals()[args.subparser](args)
