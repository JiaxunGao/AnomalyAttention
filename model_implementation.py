import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d
import random


class DeformableTokenEmbedding(nn.Module):
    """
    Deformable Token Embedding for processing sequential data with cyclic patterns.
    Treats 1D sequences as 2D inputs and applies deformable convolution to capture
    flexible temporal patterns.
    """
    def __init__(self, c_in, d_model, cycle_length):
        """
        Args:
            c_in: number of input channels
            d_model: desired output embedding dimension
            cycle_length: kernel size (and stride) in the temporal dimension,
                         which corresponds to one cycle.
        """
        super(DeformableTokenEmbedding, self).__init__()
        # Treat the 1D sequence as a 2D input with height=1 and width=L.
        # Define the offset convolution that outputs 2 * cycle_length channels (for x and y offsets).
        self.offset_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=2 * cycle_length,
            kernel_size=(1, cycle_length),
            stride=(1, cycle_length)
        )
        # Define the deformable convolution:
        self.deform_conv = DeformConv2d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=(1, cycle_length),
            stride=(1, cycle_length),
            padding=(0, 0)
        )
        # Initialize weights for both convolutions.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, L, c_in]
        Returns:
            out: Output tensor of shape [B, L_out, d_model]
        """
        # x shape: [B, L, c_in]
        # Permute to [B, c_in, L] and add a dummy height dimension → [B, c_in, 1, L]
        x = x.permute(0, 2, 1).unsqueeze(2)
        # Compute offsets: output shape [B, 2 * cycle_length, 1, L_out]
        offsets = self.offset_conv(x)
        # Apply deformable convolution with computed offsets.
        out = self.deform_conv(x, offsets)
        # Remove the height dimension and transpose → [B, L_out, d_model]
        out = out.squeeze(2).transpose(1, 2)
        return out


class GazeTransformer(nn.Module):
    """
    Transformer model for gaze pattern classification using deformable token embedding.
    Designed for processing cyclic eye movement data to classify between different conditions.
    """
    def __init__(self, circle_length, num_patches, embedding_dim, num_heads, num_layers, num_classes):
        """
        Args:
            circle_length: Length of each cycle in the input sequence
            num_patches: Number of patches after tokenization
            embedding_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
        """
        super(GazeTransformer, self).__init__()
        # Replace Conv1d patch embedding with DeformableTokenEmbedding.
        self.token_embedding = DeformableTokenEmbedding(
            c_in=3, d_model=embedding_dim, cycle_length=circle_length
        )
        
        # CLS token for classification.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        # Positional encoding for tokens, using (num_patches+1) positions to account for CLS.
        self.positional_encoding = nn.Parameter(
            self.sinusoidal_encoding(num_patches+1, embedding_dim), 
            requires_grad=False
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=num_heads, 
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def sinusoidal_encoding(self, num_positions, d_model):
        """
        Generate sinusoidal positional encodings.
        
        Args:
            num_positions: Number of positions
            d_model: Model dimension
        Returns:
            Positional encoding tensor of shape [1, num_positions, d_model]
        """
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(num_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, L, 3]
            mask: Optional attention mask
        Returns:
            output: Classification logits
            attn_weights: Attention weights from the last layer
        """
        batch_size, sequence_length, _ = x.size()
        # Tokenize using the deformable embedding.
        x = self.token_embedding(x)  # Shape: [B, L_out, embedding_dim]
        
        # Expand the CLS token and prepend it to the tokens.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, L_out+1, embedding_dim]
        
        # Add positional encoding.
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        
        # For diagnostic purposes, capture the attention weights from the last transformer layer.
        with torch.no_grad():
            temp_x = x
            for i, layer in enumerate(self.transformer.layers):
                if i == len(self.transformer.layers) - 1:  # Last layer.
                    _, attention_weights = layer.self_attn(temp_x, temp_x, temp_x)
                temp_x = layer(temp_x)
        
        # Forward pass through the full transformer.
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # CLS token output for classification.
        cls_output = x[:, 0]
        output = self.cls_head(cls_output)
        
        # Get attention weights from the last layer (for CLS token's attention).
        attn_weights = attention_weights.cpu().detach().numpy()
        
        return output, attn_weights
