from copy import deepcopy

import numpy as np
import torch


def create_data(rw, config, act_seed, validation=False):
    g, x, a = rw.run(
        config.batch_size,
        config.walk_length,
        act_seed,
        validation,
    )
    return (
        torch.tensor(x).long(),
        torch.tensor(a).long(),
        torch.tensor(g).long(),
    )


def rate_map(L, rw, model, config, device, validation=False):
    config = deepcopy(config)
    config.walk_length = config.walk_length * 50
    obs, act, pos = create_data(rw, config, 0, validation)
    chunks = [
        [i, min(i + config.tgt_len, config.walk_length)]
        for i in range(0, config.walk_length, config.tgt_len)
    ]
    config.batch_size = 1
    _obs, _act, _pos = obs[:1], act[:1], pos[:1]
    counts = np.zeros((L, L))
    pos_counts = np.zeros((L, L))
    ret = {
        "pos_emb": np.zeros((config.d_embed, L, L)),
    }
    for l in range(1, config.n_layer + 1):
        ret["layer%d/ffn_hid" % l] = np.zeros((config.d_inner, L, L))
        ret["layer%d/softmax" % l] = np.zeros(
            (config.n_head, config.mem_len + config.tgt_len + 1, L, L)
        )

    model.eval()
    with torch.no_grad():
        # Initialise the previous hidden state as none: at the beginning of a walk, there is no hidden state yet
        init_pos = _pos[:, 0].to(device)
        prev_hidden = model.rnn.init_hidden(init_pos)
        prev_outputs = None
        mems = None
        # Run through all chunks that we are going to backprop for
        for j, [start, stop] in enumerate(chunks):
            src_x = _obs[:, start:stop].to(device)
            a = _act[:, start:stop].to(device)
            if prev_outputs is not None:
                prev_hidden = model.correction(prev_hidden, prev_outputs)
            outputs = model(a, src_x, prev_hidden, mems)
            if stop >= config.mem_len + config.tgt_len:
                _x, _y = _pos[0, stop].item() % L, _pos[0, stop].item() // L
                pos_idx = _pos[0, start + 1 : stop + 1].numpy()
                pos_emb = outputs.grid_activations[0, 1:].detach().cpu().numpy()
                for n, _pos_idx in enumerate(pos_idx):
                    ret["pos_emb"][:, _pos_idx % L, _pos_idx // L] = (
                        ret["pos_emb"][:, _pos_idx % L, _pos_idx // L] + pos_emb[n]
                    )
                    pos_counts[_pos_idx % L, _pos_idx // L] = (
                        pos_counts[_pos_idx % L, _pos_idx // L] + 1
                    )
                for l in range(1, config.n_layer + 1):
                    ffn_hid = (
                        outputs.ffn_activations[l - 1][0, -1].detach().cpu().numpy()
                    )
                    ret["layer%d/ffn_hid" % l][:, _x, _y] = (
                        ret["layer%d/ffn_hid" % l][:, _x, _y] + ffn_hid
                    )
                    softmax = outputs.attentions[l - 1][0, :, -1].detach().cpu().numpy()
                    ret["layer%d/softmax" % l][..., _x, _y] = (
                        ret["layer%d/softmax" % l][..., _x, _y] + softmax
                    )
                counts[_x, _y] = counts[_x, _y] + 1
            mems = outputs.mems
            prev_hidden = outputs.rnn_hidden
            if config.correction:
                prev_outputs = outputs.last_hidden_state[:, -1].detach()

    ret["pos_emb"] = ret["pos_emb"] / pos_counts
    for l in range(1, config.n_layer + 1):
        ret["layer%d/ffn_hid" % l] = ret["layer%d/ffn_hid" % l] / counts
        ret["layer%d/softmax" % l] = ret["layer%d/softmax" % l] / counts
    return ret


def eval_memory_correct(logits, target, pos, start, stop, mlen):
    visited_pos = pos[:, max(0, start - mlen) : stop]
    c = visited_pos[..., None, None] == pos[:, stop : stop + 1]
    _visited_before = c.any(1).squeeze(-1).diagonal()
    _unvisited_before = (~c).all(1).squeeze(-1).diagonal()
    visited_correct = torch.sum(
        (
            torch.argmax(logits[_visited_before], dim=-1) == target[_visited_before]
        ).float()
    ).item()
    unvisited_correct = torch.sum(
        (
            torch.argmax(logits[_unvisited_before], dim=-1) == target[_unvisited_before]
        ).float()
    ).item()
    correct = torch.sum((torch.argmax(logits, dim=-1) == target).float()).item()
    return {
        "visited_correct": visited_correct,
        "unvisited_correct": unvisited_correct,
        "correct": correct,
        "visited_tot": _visited_before.sum(),
        "unvisited_tot": _unvisited_before.sum(),
    }
