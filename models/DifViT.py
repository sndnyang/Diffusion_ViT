import math
from .vit import *


def MLP(dim_in, dim_hidden):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_hidden),
        nn.GELU(),
        nn.LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim_hidden),
        nn.GELU(),
        nn.LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim_hidden)
    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels=3,
                 dim_head=16, dropout=0., emb_dropout=0., stochastic_depth=0.,
                 sinusoidal_cond_mlp=True, is_LSA=False, is_SPT=False, ffn_time=False):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes

        self.channels = self.out_dim = 3
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp
        time_dim = (self.num_patches + 1) * 4

        if sinusoidal_cond_mlp:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            self.time_mlp = MLP(1, time_dim)

        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_height,
                    p2=patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
        else:
            self.to_patch_embedding = ShiftedPatchTokenization(
                3, self.dim, patch_size, is_pe=True)

        self.recon_head = nn.Sequential(
            nn.Linear(self.dim, self.patch_dim),
            Rearrange(
                'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                h=image_height // patch_height,
                w=image_width // patch_width,
                p1=patch_height,
                p2=patch_width)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout,
                                       stochastic_depth, is_LSA=is_LSA, ffn_time=ffn_time)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        self.apply(init_weights)

    def forward(self, img, t=None, recon=False, feat=True):
        # patch embedding
        # print(self.patch_dim, self.dim)
        x = self.to_patch_embedding(img)

        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        if t is not None:
            t = self.time_mlp(t)
        # x = torch.cat((cls_tokens, x, t.reshape(b, -1, d)), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_emb = self.pos_embedding[:, :(n + 2)]
        x += pos_emb
        x = self.dropout(x)

        x = self.transformer(x, time_step=t)
        if recon:
            return self.recon_head(x[:, 1:])
        if feat:
            return self.mlp_head(x[:, 0])
        return self.mlp_head(x[:, 0]), self.recon_head(x[:, 1:])
