import argparse
import os.path as osp
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from configuration_transfo_xl import TransfoXLConfig
from random_walk import RandomWalker
from misc import create_data, eval_memory_correct, rate_map
from model import xlTEM

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(seed, rw, model, optimizer, lr_scheduler, criterion, config, device):
    chunks = [
        [i, min(i + config.tgt_len, config.walk_length)]
        for i in range(0, config.walk_length, config.tgt_len)
    ]
    obs, act, pos = create_data(
        rw,
        config,
        seed + 42,
    )
    # Initialise the previous hidden state as none: at the beginning of a walk, there is no hidden state yet
    init_pos = pos[:, 0].to(device)
    prev_hidden = model.rnn.init_hidden(init_pos)
    mems = None
    prev_outputs = None

    visited_tot = 0
    visited_correct = 0
    unvisited_tot = 0
    unvisited_correct = 0
    _tot = 0
    _correct = 0
    _loss = 0

    model.train()
    # Run through all chunks that we are going to backprop for
    for j, [start, stop] in enumerate(chunks):
        # Prepare data for feeding into lstm
        for i in range(start + 1, stop + 1):
            optimizer.zero_grad()
            _stop = i
            src_x = obs[:, start:_stop].to(device)
            trg_x = obs[:, _stop].to(device)
            a = act[:, start:_stop].to(device)
            if prev_outputs is not None:
                prev_hidden = model.correction(prev_hidden, prev_outputs)

            outputs = model(a, src_x, prev_hidden, mems)
            loss = criterion(outputs.logits, trg_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        mems = outputs.mems
        prev_hidden = outputs.rnn_hidden
        if config.correction:
            prev_outputs = outputs.last_hidden_state[:, -1].detach()
        with torch.no_grad():
            _loss += loss.item() * config.batch_size
            ret = eval_memory_correct(
                outputs.logits, trg_x, pos, start, stop, config.mem_len
            )
            visited_correct += ret["visited_correct"]
            unvisited_correct += ret["unvisited_correct"]
            _correct += ret["correct"]
            visited_tot += ret["visited_tot"]
            unvisited_tot += ret["unvisited_tot"]
            _tot += config.batch_size
        if lr_scheduler is not None:
            lr_scheduler.step()
    return {
        "train_tot_error": 1 - _correct / _tot,
        "train_working_memory_error": 1 - visited_correct / visited_tot,
        "train_reference_memory_error": 1 - unvisited_correct / unvisited_tot,
        "loss": _loss / _tot,
        "learning rate": config.learning_rate
        if lr_scheduler is None
        else lr_scheduler.get_last_lr()[0],
    }


def validate(seed, rw, model, config, device):
    chunks = [
        [i, min(i + config.tgt_len, config.walk_length)]
        for i in range(0, config.walk_length, config.tgt_len)
    ]
    obs, act, pos = create_data(rw, config, seed, True)

    _tot = 0
    _correct = 0
    visited_tot = 0
    visited_correct = 0
    unvisited_tot = 0
    unvisited_correct = 0

    model.eval()
    with torch.no_grad():
        # Initialise the previous hidden state as none: at the beginning of a walk, there is no hidden state yet
        init_pos = pos[:, 0].to(device)
        prev_hidden = model.rnn.init_hidden(init_pos)
        prev_outputs = None
        mems = None
        # Run through all chunks that we are going to backprop for
        for j, [start, stop] in enumerate(chunks):
            src_x = obs[:, start:stop].to(device)
            trg_x = obs[:, stop].to(device)
            a = act[:, start:stop].to(device)

            if prev_outputs is not None:
                prev_hidden = model.correction(prev_hidden, prev_outputs)
            outputs = model(a, src_x, prev_hidden, mems)
            mems = outputs.mems
            prev_hidden = outputs.rnn_hidden
            if config.correction:
                prev_outputs = outputs.last_hidden_state[:, -1].detach()

            ret = eval_memory_correct(
                outputs.logits, trg_x, pos, start, stop, config.mem_len
            )
            visited_correct += ret["visited_correct"]
            unvisited_correct += ret["unvisited_correct"]
            _correct += ret["correct"]
            visited_tot += ret["visited_tot"]
            unvisited_tot += ret["unvisited_tot"]
            _tot += config.batch_size
    return {
        "val_tot_error": 1 - _correct / _tot,
        "val_working_memory_error": 1 - visited_correct / visited_tot,
        "val_reference_memory_error": 1 - unvisited_correct / unvisited_tot,
    }


def experiment(config):
    config = TransfoXLConfig(**config)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda:%d" % config.gpu if torch.cuda.is_available() else "cpu"
    )
    config.walk_length = config.steps_per_epoch * config.tgt_len
    if config.log_to_wandb:
        run = wandb.init(
            dir=config.run_dir,
            group=f"{config.group_name}",
            project=config.proj_name,
            config=config,
        )
    rw = RandomWalker(
        config.side_len,
        config.num_envs,
        config.seed,
        config.vocab_size,
        config.n_a,
    )

    model = xlTEM(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.warmup_epoch >= 0:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.warmup_epoch * config.steps_per_epoch,
            num_training_steps=config.training_epoch * config.steps_per_epoch,
        )
    else:
        lr_scheduler = None

    t = tqdm(range(1, 1 + config.training_epoch))

    for n in t:
        train_logs = train(
            n, rw, model, optimizer, lr_scheduler, criterion, config, device
        )
        val_logs = validate(
            n,
            rw,
            model,
            config,
            device,
        )
        if config.log_to_wandb:
            wandb.log(train_logs, step=n)
            wandb.log(val_logs, step=n)

            if n % config.log_image_interval == 0:
                L = config.side_len
                ret1 = rate_map(L, rw, model, config, device)
                ret2 = rate_map(L, rw, model, config, device, True)
                m = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                for k, ret in zip(["train", "valid"], [ret1, ret2]):
                    for key, val in ret.items():
                        if "softmax" in key:
                            fig = plt.figure(figsize=(8, 8))
                            for i in range(64):
                                head = i // config.n_head
                                tmp = val[head]
                                plt.subplot(8, 8, i + 1)
                                img = m(
                                    torch.from_numpy(
                                        tmp[i % 8, :, :].reshape(1, 1, L, L)
                                    )
                                )
                                plt.imshow(img[0, 0])
                                plt.axis("off")
                            wandb.log({"%s/%s" % (k, key): fig}, step=n)
                            plt.close(fig)
                        else:
                            fig = plt.figure(figsize=(8, 8))
                            for i in range(64):
                                plt.subplot(8, 8, i + 1)
                                img = m(
                                    torch.from_numpy(val[i, :, :].reshape(1, 1, L, L))
                                )
                                plt.imshow(img[0, 0])
                                plt.axis("off")
                            wandb.log({"%s/%s" % (k, key): fig}, step=n)
                            plt.close(fig)
        train_working_memory_error = 100 * train_logs["train_working_memory_error"]
        train_reference_memory_error = 100 * train_logs["train_reference_memory_error"]
        val_working_memory_error = 100 * val_logs["val_working_memory_error"]
        t.set_description(
            desc=f"WM error: {train_working_memory_error:.3f}%, RM error {train_reference_memory_error:.3f}%, loss: {train_logs['loss']:.4f}"
        )
    if config.log_to_wandb:
        torch.save(
            {"state_dict": model.state_dict(), "maps": rw.x, "config": config},
            osp.join(run.dir, "checkpoint.pt"),
        )
        torch.save(
            {
                "train_rate_map": ret1,
                "valid_rate_map": ret2,
                "train_wm_error": train_working_memory_error,
                "train_rm_error": train_reference_memory_error,
                "valid_wm_error": val_working_memory_error,
                "config": config,
            },
            osp.join(run.dir, "rate_maps.pt"),
        )
        wandb.save("*rate_maps.pt")
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./")
    parser.add_argument("--proj_name", type=str, default="predictive coding")
    parser.add_argument("--group_name", type=str, default="Swish")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tgt_len", "-K", type=int, default=32)
    parser.add_argument("--mem_len", type=int, default=32)
    parser.add_argument("--attn_type", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--side_len", type=int, default=11)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_a", type=int, default=5)
    parser.add_argument(
        "--ffn_act_ftn",
        type=str,
        default="nmda",
        choices=["nmda", "gelu", "swish", "linear"],
    )
    parser.add_argument(
        "--rnn_act_ftn",
        type=str,
        default="tanh",
        choices=["tanh", "linear"],
    )
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.25)
    parser.add_argument("--training_epoch", type=int, default=200)
    parser.add_argument("--warmup_epoch", type=int, default=0)
    parser.add_argument("--log_image_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)

    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    experiment(config=vars(args))
