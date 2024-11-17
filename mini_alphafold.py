import torch
import torch.nn as nn
from pairformer import SimplifiedPairformer
from simplediffusion import SimplifiedDiffusion

class SimplifiedAlphaFold(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pairformer_blocks=4, diffusion_steps=100, hidden_dim=256):
        """
        Initialize the SimplifiedAlphaFold model.
        
        Args:
            vocab_size (int): Size of the amino acid vocabulary (including padding)
            embedding_dim (int): Dimension of embeddings
            pairformer_blocks (int): Number of Pairformer blocks to use
            diffusion_steps (int): Number of steps in the diffusion process
            hidden_dim (int): Dimension of hidden layers
        """
        super().__init__()
        
        # Embedding layer for amino acid sequences
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Pairformer for generating pair representations
        self.pairformer = SimplifiedPairformer(
            c=embedding_dim,
            num_blocks=pairformer_blocks
        )
        
        # Diffusion model for coordinate generation
        self.diffusion = SimplifiedDiffusion(
            embedding_dim=embedding_dim,
            num_steps=diffusion_steps
        )
        
        # MLP for processing additional properties
        self.property_mlp = nn.Sequential(
            nn.Linear(2, embedding_dim),  # 2 input properties
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Save embedding dimension for reshaping
        self.embedding_dim = embedding_dim

    def forward(self, sequence, distance_matrix, position_encoding=None, properties=None, mask=None):
        """
        Forward pass of the SimplifiedAlphaFold model.
        
        Args:
            sequence (torch.Tensor): Input amino acid sequence indices
            distance_matrix (torch.Tensor): Distance matrix between residues
            position_encoding (torch.Tensor, optional): Positional encodings
            properties (torch.Tensor, optional): Additional properties
            mask (torch.Tensor, optional): Attention mask for padding
            
        Returns:
            torch.Tensor: Predicted 3D coordinates
        """
        # Get sequence embeddings
        embeddings = self.embedding(sequence)
        
        # Add positional encoding if provided
        if position_encoding is not None:
            embeddings = embeddings + position_encoding
        
        # Process and add properties if provided
        if properties is not None:
            property_embeddings = self.property_mlp(properties)
            property_embeddings = property_embeddings.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            embeddings = torch.cat([embeddings, property_embeddings], dim=-1)
        
        # Generate pair representations using Pairformer
        pair_representation = self.pairformer(embeddings, distance_matrix, mask)
        
        # Reshape pair representation for diffusion model
        batch_size, seq_length, _ = pair_representation.shape
        pair_representation = pair_representation.view(batch_size, seq_length, self.embedding_dim)
        
        # Generate coordinates using diffusion model
        predicted_coords = self.diffusion.sample(pair_representation)
        
        return predicted_coords

    def training_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch (dict): Dictionary containing:
                - sequence: Amino acid sequence indices
                - distance_matrix: Distance matrix between residues
                - coordinates: True 3D coordinates
                - position_encoding (optional): Positional encodings
                - properties (optional): Additional properties
                
        Returns:
            torch.Tensor: Loss value
        """
        # Move batch data to device
        sequence = batch['sequence']
        distance_matrix = batch['distance_matrix']
        true_coords = batch['coordinates']
        
        # Handle optional inputs
        position_encoding = batch.get('position_encoding')
        properties = batch.get('properties')
        
        # Create mask based on padding
        mask = sequence == 0
        
        # Forward pass
        predicted_coords = self(
            sequence,
            distance_matrix,
            position_encoding,
            properties,
            mask
        )
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_coords, true_coords)
        
        return loss


def create_model(vocab_size, embedding_dim, device=None):
    """
    Helper function to create and initialize the model.
    
    Args:
        vocab_size (int): Size of amino acid vocabulary
        embedding_dim (int): Dimension of embeddings
        device (torch.device, optional): Device to place model on
        
    Returns:
        SimplifiedAlphaFold: Initialized model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = SimplifiedAlphaFold(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    ).to(device)
    
    return model


# # Example usage:
# if __name__ == "__main__":
#     # Model parameters
#     VOCAB_SIZE = 6  # Size of amino acid vocabulary + padding
#     EMBEDDING_DIM = 32
    
#     # Create model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = create_model(VOCAB_SIZE, EMBEDDING_DIM, device)
    
#     # Initialize optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
#     # Training loop
#     NUM_EPOCHS = 100
#     for epoch in range(NUM_EPOCHS):
#         # Replace this with your actual data loading
#         batch = next(iter(train_dataloader))  # Your dataloader here
        
#         # Training step
#         optimizer.zero_grad()
#         loss = model.training_step(batch)
#         loss.backward()
#         optimizer.step()
        
#         print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


    