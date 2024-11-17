import torch
import torch.nn as nn
import numpy as np

def triangle_update(pair_rep, single_rep, outgoing=True):
    """
    Performs a triangle update on the pair representation.
    """
    n = pair_rep.size(0)
    updated_pair_rep = torch.zeros_like(pair_rep)
    
    if outgoing:
        # Updating along rows (outgoing updates)
        for i in range(n):
            updated_values = single_rep.clone()
            if i > 0:
                updated_values += pair_rep[i, :i].sum(dim=0)
            updated_pair_rep[i] = updated_values.unsqueeze(0)
    else:
        # Updating along columns (incoming updates)
        for j in range(n):
            updated_values = single_rep.clone()
            if j > 0:
                updated_values += pair_rep[:j, j].sum(dim=0)
            updated_pair_rep[:, j] = updated_values.unsqueeze(0)
            
    return updated_pair_rep

class TriangleAttention(nn.Module):
    def __init__(self, c, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.c = c
        self.head_dim = c // num_heads
        assert self.head_dim * num_heads == c, "c must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(c, 3 * c)
        self.out_proj = nn.Linear(c, c)
        
        # Optional learnable scaling for distance matrix
        self.distance_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, pair_rep, mask=None, distance_matrix=None):
        n = pair_rep.size(0)
        
        # Project to Q, K, V
        qkv = self.qkv_proj(pair_rep)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(n, n, self.num_heads, self.head_dim).permute(2, 0, 1, 3)  # (heads, n, n, head_dim)
        k = k.view(n, n, self.num_heads, self.head_dim).permute(2, 0, 3, 1)  # (heads, n, head_dim, n)
        v = v.view(n, n, self.num_heads, self.head_dim).permute(2, 0, 1, 3)  # (heads, n, n, head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k) / np.sqrt(self.head_dim)  # (heads, n, n, n)
        
        # Add distance information to attention scores
        if distance_matrix is not None:
            # Scale distances with learnable parameter
            scaled_distances = self.distance_scale * distance_matrix
            # Reshape distance matrix for broadcasting
            scaled_distances = scaled_distances.unsqueeze(0).expand(self.num_heads, -1, -1)
            attention_scores = attention_scores + scaled_distances.unsqueeze(1)
            
        if mask is not None:
            # Expand mask for heads dimension
            mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            
        # Apply softmax and compute weighted sum
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, v)
        
        # Reshape back to original dimensions
        attended_values = attended_values.permute(1, 2, 0, 3).contiguous().view(n, n, self.c)
        
        return self.out_proj(attended_values)

class SimplifiedPairformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        # Triangle attention modules
        self.triangle_attention_row = TriangleAttention(c)
        self.triangle_attention_col = TriangleAttention(c)
        
        # Layer normalization
        self.single_norm = nn.LayerNorm(c)
        self.pair_norm_row = nn.LayerNorm(c)
        self.pair_norm_col = nn.LayerNorm(c)
        
        # MLP for transitions
        self.linear1 = nn.Linear(c, 2*c)
        self.linear2 = nn.Linear(2*c, c)
        self.relu = nn.ReLU()
        
    def forward(self, pair_rep, single_rep, mask=None, distance_matrix=None):
        # Row-wise updates
        pair_rep = self.pair_norm_row(pair_rep + triangle_update(pair_rep, single_rep))
        pair_rep = self.pair_norm_row(pair_rep + 
                                    self.triangle_attention_row(pair_rep, mask, distance_matrix))
        
        # Column-wise updates
        pair_rep = self.pair_norm_col(pair_rep + triangle_update(pair_rep, single_rep, outgoing=False))
        pair_rep = self.pair_norm_col(pair_rep + 
                                    self.triangle_attention_col(pair_rep, 
                                    mask.transpose(0, 1) if mask is not None else None,
                                    distance_matrix))
        
        # Single representation update
        single_rep = self.single_norm(single_rep + 
                    self.relu(self.linear2(self.relu(self.linear1(single_rep)))))
        
        return pair_rep, single_rep

class SimplifiedPairformer(nn.Module):
    def __init__(self, c, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([SimplifiedPairformerBlock(c) for _ in range(num_blocks)])
        
    def forward(self, single_rep, distance_matrix=None, mask=None):
        """
        Args:
            single_rep: (torch.Tensor) Single representation (N, C)
            distance_matrix: (torch.Tensor, optional) Distance matrix (N, N)
            mask: (torch.Tensor, optional) Attention mask (N, N)
        """
        n = single_rep.size(0)
        
        # Initialize pair representation
        pair_rep = single_rep.unsqueeze(1) + single_rep.unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.blocks:
            pair_rep, single_rep = block(pair_rep, single_rep, mask, distance_matrix)
            
        return pair_rep

def example_usage():
    # Parameters
    seq_length = 32
    hidden_dim = 64
    
    # Create model
    model = SimplifiedPairformer(c=hidden_dim, num_blocks=4)
    
    # Create dummy input
    single_rep = torch.randn(seq_length, hidden_dim)
    distance_matrix = torch.randn(seq_length, seq_length)
    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
    
    # Forward pass
    output = model(single_rep, distance_matrix, mask)
    print(f"Output shape: {output.shape}")  # Should be (seq_length, seq_length, hidden_dim)

if __name__ == "__main__":
    example_usage()


# To use this implementation, you can create an instance of SimplifiedPairformer with your desired hidden dimension size (c) and number of blocks. The example usage at the bottom shows how to create and use the model.
# A few important notes:

# The hidden dimension (c) must be divisible by the number of attention heads (default is 4).
# The model can handle optional distance matrices and attention masks.
# The implementation includes all essential components while maintaining readability and efficiency.


# Key improvements:

# More principled handling of distance information by incorporating it into the attention mechanism
# Learnable scaling of distance information
# Better control over how distance information influences the attention weights
# Maintained proper dimensionality throughout the attention computation

# The distance matrix now directly influences the attention weights rather than being added to the pair representation. This allows the model to learn how to use the distance information more effectively during the attention computation.