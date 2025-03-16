import torch
from torch import nn

from timm.layers import use_fused_attn
from typing import Type, Optional

from torch.nn import functional as F


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            dropout: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head self-attention for the input tensor, optionally using an attention mask.

        This method projects the input tensor into query, key, and value representations, applies
        multi-head scaled dot-product attention (using either a fused or manual implementation), and
        projects the result back to the original dimensionality.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, N, C) where:
                - B is the batch size,
                - N is the sequence length,
                - C is the number of features (embedding dimension).
            attn_mask (Optional[torch.Tensor]): An optional attention mask to prevent attention to certain positions.
                The mask can be of shape (L, S) or (B * num_heads, L, S), where:
                - L is the target sequence length,
                - S is the source sequence length.
                For a boolean mask, a True value indicates positions that should be attended to.
                For a float mask, the mask values are added to the attention logits.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x (torch.Tensor): Output tensor after applying attention, with shape (B, N, C).
                - attn (torch.Tensor): The attention weights computed during the process.
        
        Notes:
            - When fused attention is enabled (and supported), the function leverages PyTorch's
            `F.scaled_dot_product_attention` and passes the attention mask directly.
            - In the manual implementation, the attention logits are computed and scaled. If an
            attention mask is provided:
                - For boolean masks, positions where the mask is False are inverted via `logical_not`
                    and set to -infinity to zero out their contribution after softmax.
                - For float masks, the mask is added directly to the logits.
            - After applying the softmax and dropout, the attention weights are used to compute a weighted
            sum of the values.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # If using fused attention, pass the mask to the PyTorch function (if supported)
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )
            attn = None
        else:
            raise NotImplementedError("Non-Fused attention not implemented yet")
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                print(f"attn_mask: {attn_mask.shape}", f"attn: {attn_mask}")
                # If the mask is boolean, mask out positions by setting them to -inf
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(attn_mask.logical_not(), float('-inf'))
                else:
                    # For a float mask, add the mask to the logits directly
                    raise NotImplementedError("Float mask not implemented yet")
                    attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., qk_norm=False):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout, qk_norm=qk_norm)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for idx, (attn, ff) in enumerate(self.layers):
            attention_value, attention_weight = attn(x, attn_mask=attn_mask)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn

class MaskTransformer(nn.Module):
    def __init__(self, img_size=28, num_tokens= 8 ,hidden_dim=512, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=2, ignore_attn_to_source=False, qk_norm=False):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
        """

        super().__init__()
        self.nclass = nclass
        self.num_tokens = num_tokens
        self.patch_size = img_size // num_tokens
        self.codebook_size = codebook_size
        self.class_emb = nn.Embedding(nclass, hidden_dim)  # +1 for the mask of the viz token, +1 for mask of the class
        self.tok_emb = nn.Linear(hidden_dim, hidden_dim) 
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.num_tokens*self.num_tokens)+3 +1 , hidden_dim)), 0., 0.02) #+1 for time
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, qk_norm=qk_norm)
        
        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # Bias for the last linear output
        #self.bias = nn.Parameter(torch.zeros((self.num_tokens*self.num_tokens)+3 +1 , codebook_size+1+nclass+1))#(self.num_tokens*self.num_tokens)+3, codebook_size+1+nclass+1))
        self.ignore_attn_to_source = ignore_attn_to_source

    def forward(self, img_token, t, y=None ,drop_label=None, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, d, w, h= img_token.size()
        #print(f"y: {y}")
        y = torch.stack([i for i in y ], dim= 1) if y is not None else None
        #print(f"y after: {y.shape}")
        if y is not None:
            #print(f"y shape: {y.shape}")
            cls_token = y.view(b, -1) #+ self.codebook_size + 1  # Shift the class token by the amount of codebook # +1 not 33 
            if drop_label is not None:
                cls_token[drop_label.long()] = self.codebook_size + 1 + self.nclass  # Drop condition
            #input = torch.cat([img_token.view(b, -1), cls_token.view(b, -1)], -1)  # concat visual tokens and class tokens
        else:
            input = img_token.view(b, -1)
        #print(f"input size:{input.shape}, img_token: {img_token.size()}")
        # film conditiong , or add it to the d ,
        
        input = img_token.permute(0, 2, 3, 1).reshape(b, w*h, d)  # b, w*h, d
                
        tok_embeddings = self.tok_emb(input)
        cls_emb = self.class_emb(cls_token)
        t = t.view(b, 1) 
        t_emb = self.time_emb(t)  
        #t_emb = t_emb.unsqueeze(1).expand(-1, tok_embeddings.shape[1], -1)  
        t_emb = t_emb.unsqueeze(1)
        #cls_emb = cls_emb.unsqueeze(1)
        #print(f"cls_emb: {cls_emb.shape}, t_emb: {t_emb.shape}, tok: {tok_embeddings.shape}")
        all_emb = torch.cat([tok_embeddings ,cls_emb, t_emb], 1)
        
        # Position embedding
        pos_embeddings = self.pos_emb
        #print(f"pos_emb :{pos_embeddings.shape}, tok: {all_emb.shape}")
        
        x = all_emb + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x) if not self.ignore_attn_to_source else self.transformer(x, attn_mask=self.build_ignore_source_mask(x.size(1), w*h, device=x.device))
        x = x[:, : w*h, :]
        # premuate x to b, d, w, h
        b, hw, d = x.shape
        # change ([64, 128, 64]) to ([64, 128, 8,8])
        c_h = int(hw ** 0.5)
        x = x.permute(0, 2, 1)
        x = x.reshape(b, d, c_h, c_h)
        

        return x 
        #return logit[:, :self.num_tokens*self.num_tokens, :self.codebook_size+1]
    
    def build_ignore_source_mask(self, total_tokens: int, num_img_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Build an attention mask of shape (total_tokens, total_tokens) such that:
        - For query positions corresponding to image tokens (indices 0 to num_img_tokens-1), no masking is applied.
        - For query positions corresponding to label tokens (indices num_img_tokens to total_tokens-1), 
            all keys corresponding to image tokens are masked (set to -inf) and, optionally, label-to-label attention
            is disabled except for the diagonal (self-attention) so that label tokens remain unchanged.
        """

        # Initialize the mask with all True values.
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)
        
        # For label token queries (rows from num_img_tokens onward), disallow attention to image tokens.
        mask[num_img_tokens:, :num_img_tokens] = False
        
        # For label token queries, disallow attention to other label tokens by default.
        mask[num_img_tokens:, num_img_tokens:] = False
        
        return mask

