import os.path as osp

import torch
import torch.nn as nn
import pandas as pd
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model



class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_prompt=3):
        super().__init__()
        self.n_prompts = n_prompt
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = 'a video of action'
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_prompts, n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand((self.n_prompts, self.n_cls) + ctx.size())

        # prefix = self.token_prefix
        # suffix = self.token_suffix
        prefix = self.token_prefix.unsqueeze(0).expand((self.n_prompts, self.n_cls) + self.token_prefix.size()[1:])
        suffix = self.token_suffix.unsqueeze(0).expand((self.n_prompts, self.n_cls) + self.token_suffix.size()[1:])
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_prompts, n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=2,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



class TextEncoder(nn.Module):
    def __init__(self, classnames, clip_model, n_prompt):
        super().__init__()
        self.transformer = clip_model.transformer
        self.prompt_learner = PromptLearner(classnames, clip_model, n_prompt)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text_id,  list_id):
        prompts = self.prompt_learner()
        prompts = torch.stack([prompts[j][i] for i,j in zip(list_id,text_id)])
        tokenized_prompts = self.tokenized_prompts[list_id]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x.type(self.transformer.resblocks[0].mlp.c_fc.weight.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_sequence = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x_sequence.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x, x_sequence



if __name__ == '__main__':
    file = pd.read_csv('./lists/minikinetics_400_labels.csv')
    class_names = []
    for idx in range(len(file)):
        class_names.append(file.iloc[idx, 1])

    n_prompts = 3
    clip_model, clip_state_dict = clip.load('ViT-B/32',device='cpu',jit=False, tsm=False, T=8,dropout=0.0, emb_dropout=0.0,pretrain=True, joint=False) #Must set jit=False for training  ViT-B/32
    net = TextEncoder(class_names, clip_model, n_prompts).cuda()

    fake_data = torch.randn(2, 3, 224, 224).cuda()
    list_id = torch.LongTensor([2, 3])
    text_id = np.random.randint(n_prompts, size=len(list_id))
    output, x_sequence = net(text_id, list_id)
    print(output)
    