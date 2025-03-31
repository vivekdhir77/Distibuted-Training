import torch
import torch.nn as nn

class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = n_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, embed_dim)  # Linear projection
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, -1)  # Flatten patches
        x = self.proj(x)  # Linear projection
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        return self.dropout(self.norm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path1 = nn.Dropout(stochastic_depth)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, forward_expansion * embed_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(forward_expansion * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path2 = nn.Dropout(stochastic_depth)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.dropout(self.fc2(self.activation(self.fc1(self.norm2(x))))))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, num_layers, num_heads, forward_expansion, image_size, patch_size, num_classes, dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout)
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, forward_expansion, dropout, stochastic_depth)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x[:, 0])  # Use CLS token
        return x