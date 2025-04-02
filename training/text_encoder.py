import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from typing import Union, List
from pkg_resources import packaging

from text_encoder_utils import SimpleTokenizer as _Tokenizer


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    _tokenizer = _Tokenizer()

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, maple_prompts=None):
        if maple_prompts:
            num_prompts = maple_prompts[0].shape[0]
            for i, blk in enumerate(self.resblocks):
                if i == 0:
                    x = blk(x)
                else:
                    prefix = x[:1, :, :]
                    suffix = x[1 + num_prompts:, :, :]
                    # Create/configure learnable tokens of this layer
                    textual_context = maple_prompts[i - 1]
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    # Add the learnable tokens of this layer with the input, replaced by previous
                    # layer learnable tokens
                    x = torch.cat([prefix, textual_context, suffix], dim=0)

                    # then do forward pass from transformer
                    x = blk(x)
        else:
            for blk in self.resblocks:
                x = blk(x)
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int = 512,
            context_length: int = 77,
            vocab_size: int = 49408,
            transformer_width: int = 512,
            transformer_heads: int = 8,
            transformer_layers: int = 12,
    ):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, prompts, tokenized_prompts, maple_prompts=None):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        if maple_prompts:
            x = self.transformer(x, maple_prompts)
        else:
            x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] # Dimension of ctx
        vis_dim = clip_model.visual.output_dim      # Dimension of image summary token (s)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # --- Initialize original learnable context tokens (ctx) ---
        if ctx_init:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # Register original ctx as learnable parameter
        self.ctx = nn.Parameter(ctx_vectors) # Shape: (n_ctx, ctx_dim)

        # --- Define the attention scoring layer (replaces meta_net) ---
        # Linear layer W for score: tanh(W[s; ctx])
        # Input: concatenated features, Output: 1 (scalar score)
        # add bias
        self.attention_scorer = nn.Linear(vis_dim + ctx_dim, 1)


        # --- Prepare classnames and fixed tokens (SOS, CLS, EOS) ---
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # Unused? Keep for now.
        # Base prompts used only to get fixed embeddings
        prompts_for_embedding = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_for_embedding])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Register fixed parts as buffers (non-learnable but part of state)
        self.register_buffer("token_prefix", embedding[:, :1, :])          # SOS, Shape: (n_cls, 1, ctx_dim)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) # CLS & EOS, Shape: (n_cls, *, ctx_dim)

        # Store constants
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts # Token indices, needed by TextEncoder
        # self.ctx_dim = ctx_dim # Redundant, can use self.ctx.shape[-1]
        # self.vis_dim = vis_dim # Redundant, can use im_features.shape[-1] in forward

    # construct_prompts method concatenates prefix, ctx, suffix
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # Optional label-based selection (not used in this forward pass)
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        # Concatenate along token dimension (dim=1)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    # --- Forward pass implementing the new attention-based update ---
    def forward(self, im_features):
        """
        Updates context tokens ctx based on image features (s) via attention,
        then constructs final prompts for all classes.

        Args:
            im_features (torch.Tensor): Image features (s). Shape: (batch_size, vis_dim).

        Returns:
            torch.Tensor: Final prompt embeddings for all classes per image.
                          Shape: (batch_size, n_cls, total_tokens, ctx_dim).
        """
        prefix = self.token_prefix       # Shape: (n_cls, 1, ctx_dim)
        suffix = self.token_suffix       # Shape: (n_cls, *, ctx_dim)
        ctx_original = self.ctx          # Shape: (n_ctx, ctx_dim)

        batch_size = im_features.shape[0]
        n_ctx = self.n_ctx
        ctx_dim = self.ctx.shape[-1]     # Get ctx dimension dynamically

        # Prepare s and ctx for pairwise scoring
        # Expand s per ctx token: (batch_size, n_ctx, vis_dim)
        s_expanded = im_features.unsqueeze(1).expand(-1, n_ctx, -1)
        # Expand ctx per batch item: (batch_size, n_ctx, ctx_dim)
        ctx_expanded = ctx_original.unsqueeze(0).expand(batch_size, -1, -1)

        # --- Step 1: Calculate scores for all (s, ctx) pairs ---
        # Concatenate features: (batch_size, n_ctx, vis_dim + ctx_dim)
        concat_features = torch.cat([s_expanded, ctx_expanded], dim=-1)
        # Apply linear layer W: (batch_size, n_ctx, 1)
        scores_raw = self.attention_scorer(concat_features)
        # Apply tanh activation: (batch_size, n_ctx, 1)
        scores_activated = torch.tanh(scores_raw)
        # Remove trailing dim: (batch_size, n_ctx)
        scores = scores_activated.squeeze(-1)

        # --- Step 2: Calculate attention weights 'a' ---
        # Softmax scores along ctx dimension (dim=1) for each image
        # attention_weights shape: (batch_size, n_ctx)
        attention_weights = F.softmax(scores, dim=1)

        # --- Step 3 & 4: Calculate aggregated context update 'C' ---
        # Weighted sum of ctx tokens using attention weights
        # Reshape weights for broadcasting: (batch_size, n_ctx, 1)
        # C = sum(a_i * ctx_i) over i
        # context_update_C shape: (batch_size, ctx_dim)
        context_update_C = torch.sum(attention_weights.unsqueeze(-1) * ctx_expanded, dim=1)

        # --- Step 5: Update ctx ---
        # Add the aggregated context C to the original ctx for each batch item
        # Use broadcasting: (1, n_ctx, ctx_dim) + (batch_size, 1, ctx_dim)
        # ctx_shifted shape: (batch_size, n_ctx, ctx_dim)
        ctx_shifted = ctx_original.unsqueeze(0) + context_update_C.unsqueeze(1)

        # --- Step 6: Construct final prompts for all classes ---
        # Initialize list to store prompts for each image
        prompts = []
        # Iterate through the batch dimension of updated contexts
        for i in range(batch_size):
            # Get updated context for the i-th image: (n_ctx, ctx_dim)
            ctx_shifted_i = ctx_shifted[i]
            # Expand context for all classes: (n_cls, n_ctx, ctx_dim)
            ctx_i_expanded = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            # Construct prompts for this image (all classes): (n_cls, total_tokens, ctx_dim)
            pts_i = self.construct_prompts(ctx_i_expanded, prefix, suffix)
            prompts.append(pts_i)

        # Stack prompts along the batch dimension
        # Final prompts shape: (batch_size, n_cls, total_tokens, ctx_dim)
        prompts = torch.stack(prompts)

        return prompts


class TextPromptLearnerWithoutSummary(nn.Module):
    def __init__(self, classnames, text_model, num_prompts, prompts_init='', CSC=False, ctx_pos='end'):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = len(classnames)
        n_ctx = num_prompts
        ctx_init = prompts_init
        ctx_dim = text_model.ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = text_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        # print(tokenized_prompts.shape)
        with torch.no_grad():
            embedding = text_model.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_pos

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])