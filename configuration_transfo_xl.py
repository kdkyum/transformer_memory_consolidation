class TransfoXLConfig:
    model_type = "transfo-xl"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=10,
        n_a=5,
        rnn_act_ftn="tanh",
        d_rnn=128,
        d_model=512,
        d_embed=256,
        n_head=8,
        d_head=64,
        d_inner=2048,
        ffn_act_ftn="relu",
        alpha=1.0,
        beta=1.0,
        div_val=4,
        pre_lnorm=True,
        n_layer=1,
        mem_len=0,
        ext_len=0,
        tgt_len=0,
        clamp_len=0,
        same_length=False,
        attn_type=1,
        dropout=0.1,
        dropatt=0.0,
        untie_r=True,
        correction=False,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        side_len=11,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_a = n_a
        self.rnn_act_ftn = rnn_act_ftn
        self.d_rnn = d_rnn
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_head = d_head
        self.d_inner = d_inner
        self.div_val = div_val
        self.pre_lnorm = pre_lnorm
        self.n_layer = n_layer
        self.n_head = n_head
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.tgt_len = tgt_len
        self.same_length = same_length
        self.attn_type = attn_type
        self.clamp_len = clamp_len
        self.dropout = dropout
        self.dropatt = dropatt
        self.untie_r = untie_r
        self.correction = correction
        self.init = init
        self.init_range = init_range
        self.proj_init_std = proj_init_std
        self.init_std = init_std
        self.layer_norm_epsilon = layer_norm_epsilon
        self.ffn_act_ftn = ffn_act_ftn
        self.alpha = alpha
        self.beta = beta
        self.side_len = side_len
        self.__dict__.update(kwargs)
