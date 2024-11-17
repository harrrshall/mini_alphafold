import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np

class DenoisingNetwork(nn.Module):
    """
    Neural network that predicts noise at each timestep of the diffusion process.
    
    Args:
        input_dim (int): Dimension of input (coordinates + timestep embedding + sequence embedding)
        hidden_dim (int): Dimension of hidden layers
        output_dim (int, optional): Dimension of output (default: 2 for 2D coordinates)
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predicts the noise added at timestep t.
        
        Args:
            x_t: Noisy data at timestep t (batch_size, N, 2)
            t: Timestep indices (batch_size,)
            sequence_embeddings: Sequence embeddings (batch_size, N, embedding_dim)
            
        Returns:
            torch.Tensor: Predicted noise (batch_size, N, 2)
        """
        t_embedding = self.timestep_embedding(t)
        t_embedding = t_embedding.unsqueeze(1).expand(-1, x_t.size(1), -1)
        input_data = torch.cat([x_t, t_embedding, sequence_embeddings], dim=-1)
        return self.mlp(input_data)

    def timestep_embedding(self, timesteps: torch.Tensor, dim: int = 128, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: Timestep indices (batch_size,)
            dim: Embedding dimension (default: 128)
            max_period: Controls minimum frequency of embeddings (default: 10000)
            
        Returns:
            torch.Tensor: Timestep embeddings (batch_size, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class SimplifiedDiffusion(nn.Module):
    """
    Simplified Diffusion model for generating 2D coordinates.
    
    Args:
        embedding_dim (int): Dimension of sequence embeddings
        num_steps (int, optional): Number of diffusion steps (default: 100)
        beta_start (float, optional): Starting noise level (default: 1e-4)
        beta_end (float, optional): Ending noise level (default: 0.02)
    """
    def __init__(self, embedding_dim, num_steps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        
        self.denoising_network = DenoisingNetwork(
            input_dim=embedding_dim + 2 + 128,  # 2 for coordinates, 128 for timestep embedding
            hidden_dim=256
        )
        
        # Register diffusion parameters as buffers
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bar', torch.cumprod(self.alphas, dim=0))

    def sample(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D coordinates using the reverse diffusion process.
        
        Args:
            sequence_embeddings: Sequence embeddings (batch_size, N, embedding_dim)
            
        Returns:
            torch.Tensor: Generated 2D coordinates (batch_size, N, 2)
        """
        batch, n, _ = sequence_embeddings.shape
        device = sequence_embeddings.device
        
        # Start with random noise
        x = torch.randn(batch, n, 2, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_steps)):
            # No noise at final step
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            # Create timestep tensor
            timesteps = torch.ones((batch,), device=device) * t
            
            # Predict noise
            predicted_noise = self.denoising_network(x, timesteps, sequence_embeddings)
            
            # Get diffusion parameters
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            
            # Update x using reverse diffusion formula
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(self.betas[t]) * z
            
        return x

    def training_step(self, x: torch.Tensor, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            x: Input coordinates (batch_size, N, 2)
            sequence_embeddings: Sequence embeddings (batch_size, N, embedding_dim)
            
        Returns:
            torch.Tensor: Loss value
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Sample random timesteps for each element in the batch
        t = torch.randint(0, self.num_steps, (batch,), device=device)

        # Add noise to the input, handling timesteps correctly
        noise = torch.randn_like(x)
        
        # Gather alpha_bar values for each timestep in the batch
        alpha_bar_t = self.alpha_bar[t].view(batch, 1, 1).expand(-1, seq_len, x.shape[2])
        
        # Create noisy samples using the correct alpha_bar values
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1.0 - alpha_bar_t) * noise

        # Predict noise
        predicted_noise = self.denoising_network(noisy_x, t, sequence_embeddings)

        # Calculate loss
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss

def main():
    """Example usage of SimplifiedDiffusion"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    batch_size = 8
    seq_length = 10
    embedding_dim = 16
    num_steps = 100
    
    # Create dummy data
    sequence_embeddings = torch.randn(batch_size, seq_length, embedding_dim)
    coordinates = torch.randn(batch_size, seq_length, 2)
    
    # Initialize model and optimizer
    model = SimplifiedDiffusion(embedding_dim=embedding_dim, num_steps=num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model.training_step(coordinates, sequence_embeddings)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Generate new coordinates
    with torch.no_grad():
        generated_coords = model.sample(sequence_embeddings)
    
    # Visualize results
    plt.figure(figsize=(10, 5))
    
    # Plot original coordinates
    plt.subplot(1, 2, 1)
    for i in range(batch_size):
        plt.scatter(coordinates[i, :, 0].numpy(), coordinates[i, :, 1].numpy(), label=f'Sequence {i+1}')
    plt.title('Original Coordinates')
    plt.legend()
    
    # Plot generated coordinates
    plt.subplot(1, 2, 2)
    for i in range(batch_size):
        plt.scatter(generated_coords[i, :, 0].numpy(), generated_coords[i, :, 1].numpy(), label=f'Sequence {i+1}')
    plt.title('Generated Coordinates')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()