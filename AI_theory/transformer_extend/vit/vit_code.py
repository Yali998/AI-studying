import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, 
                 num_classes = 1000, dim = 768, depth = 12, heads = 12,
                 mlp_dim = 3072, dropout = 0.1):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        # self.patch_to_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_to_embedding = nn.Linear(patch_size * patch_size * 3, dim)

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, dim)) # +1 for cls token

        self.transformer = nn.Transformer(dim, heads, depth, depth, dim_feedforward=mlp_dim, dropout=dropout)
        self.mlp_head = nn.Linear(dim, num_classes)  # final classification layer

    def forward(self, x):
        '''
        Args:
            x: input tensor of shape [batch_size, channels, height, width]
        Returns:
            logits: output tensor of shape [batch_size, num_classes]
        '''
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, 
                   W // self.patch_size, self.patch_size)  # [B, C, H/P, P, W/P, P])
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, H/P, W/P, P, P, C])
        x = x.view(B, self.num_patches, -1) # [B, num_patches, P*P*C] patches flatten
        x = self.patch_to_embedding(x)  # [B, num_patches, dim]

        # add cls token, positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [1, 1, dim] -> [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim = 1) # [B, num_patches + 1, dim]
        x += self.position_embeddings  # [B, num_patches + 1, dim]

        # transformer
        x = x.permute(1, 0, 2)  # [num_patches + 1, B, dim] for transformer input)
        x = self.transformer(x)
        x = x[0]  # take cls token output [B, dim]

        x = self.mlp_head(x)
        return x
    
# test
model = VisionTransformer()
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(output.shape)  # should be [1, num_classes]