import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from minifold_dataset import EnhancedAminoAcidAlphabet

class AminoAcid:
    __slots__ = ['code', 'hydrophobicity', 'size', 'charge', 'secondary_structure_preference']
    code: str
    hydrophobicity: float
    size: float
    charge: float
    secondary_structure_preference: str

class PreprocessedProteinDataset(Dataset):
    PAD_TOKEN = "<PAD>"  # Define padding token
    
    def __init__(self, data: List[Dict], alphabet, include_relative_encoding: bool = True, 
                 preprocessed_dir: Optional[str] = None, max_seq_length: Optional[int] = None):
        self.data = data
        self.alphabet = alphabet
        # Add padding token to alphabet if it doesn't exist
        if not hasattr(self.alphabet, 'padding_idx'):
            if self.PAD_TOKEN not in self.alphabet.aa_to_idx:
                self.alphabet.aa_to_idx[self.PAD_TOKEN] = len(self.alphabet.aa_to_idx)
            self.alphabet.padding_idx = self.alphabet.aa_to_idx[self.PAD_TOKEN]
        
        self.include_relative_encoding = include_relative_encoding
        self.preprocessed_dir = preprocessed_dir
        self.max_seq_length = max_seq_length or self._calculate_max_length()
        self._cache = {}
        self.dataset_stats = None
        
        # Load preprocessed data if available
        if preprocessed_dir and os.path.exists(preprocessed_dir):
            self._load_preprocessed_data()

    def __len__(self) -> int:
        """Return the number of items in the dataset"""
        return len(self.data)

    def _calculate_max_length(self) -> int:
        """Calculate the maximum sequence length in the dataset"""
        return max(item['length'] for item in self.data)

    def _load_preprocessed_data(self):
        """Load preprocessed data from disk"""
        # Load dataset statistics
        stats_path = os.path.join(self.preprocessed_dir, 'dataset_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.dataset_stats = json.load(f)

        # Load preprocessed tensors
        for idx in range(len(self)):
            tensor_path = os.path.join(self.preprocessed_dir, f'protein_{idx}.pt')
            if os.path.exists(tensor_path):
                self._cache[idx] = torch.load(tensor_path)

    def compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix from 2D coordinates."""
        num_residues = coords.shape[0]
        dist_matrix = np.zeros((num_residues, num_residues))
        
        for i in range(num_residues):
            for j in range(i + 1, num_residues):
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                
        return dist_matrix

    def compute_relative_position_encoding(self, dist_matrix: np.ndarray, 
                                        num_frequencies: int = 8,
                                        max_distance: Optional[float] = None) -> np.ndarray:
        """
        Compute relative positional encodings using sine and cosine functions.
        Returns: Array of shape (N, N, 2 * num_frequencies) where N is sequence length
        """
        if max_distance is None and self.dataset_stats:
            max_distance = self.dataset_stats['distance_stats']['global_stats']['max_distance']
        elif max_distance is None:
            max_distance = 20.0  # fallback default
            
        seq_length = dist_matrix.shape[0]
        pos_enc = np.zeros((seq_length, seq_length, 2 * num_frequencies))
        
        # Normalize distances
        dist_matrix = np.clip(dist_matrix, 0, max_distance) / max_distance
        
        for i in range(num_frequencies):
            freq = 2.0 ** i
            pos_enc[:, :, 2*i] = np.sin(dist_matrix * freq * np.pi)
            pos_enc[:, :, 2*i+1] = np.cos(dist_matrix * freq * np.pi)
            
        return pos_enc

    def pad_sequence(self, sequence: torch.Tensor, max_length: int) -> torch.Tensor:
        """Pad sequence to max_length"""
        padding_length = max_length - len(sequence)
        if padding_length > 0:
            padding = torch.full((padding_length,), self.alphabet.padding_idx, dtype=sequence.dtype)
            return torch.cat([sequence, padding])
        return sequence

    def pad_matrix(self, matrix: torch.Tensor, max_length: int, pad_value: float = 0.0) -> torch.Tensor:
        """Pad 2D or 3D matrix to max_length in first two dimensions"""
        current_shape = matrix.shape
        if len(current_shape) == 2:
            padding = (0, max_length - current_shape[1], 0, max_length - current_shape[0])
            return torch.nn.functional.pad(matrix, padding, value=pad_value)
        else:  # 3D tensor for position encoding
            padding = (0, 0, 0, max_length - current_shape[1], 0, max_length - current_shape[0])
            return torch.nn.functional.pad(matrix, padding, value=pad_value)

    def normalize_features(self, item: Dict) -> Dict:
        """Normalize features using dataset statistics"""
        if not self.dataset_stats:
            return item

        stats = self.dataset_stats
        
        # Normalize distance matrix if it exists in item
        if 'distance_matrix' in item:
            dist_stats = stats['distance_stats']['global_stats']
            item['distance_matrix'] = (item['distance_matrix'] - dist_stats['mean_distance']) / dist_stats['std_distance']
        
        # Normalize properties if they exist
        if 'properties' in item and 'property_stats' in stats:
            prop_stats = stats['property_stats']
            if isinstance(item['properties'], torch.Tensor):
                props = item['properties'].tolist()
            else:
                props = item['properties']
                
            normalized_props = []
            property_names = ['hydrophobic_ratio', 'charged_ratio']
            for i, prop in enumerate(property_names):
                if isinstance(props, dict):
                    prop_val = props[prop]
                else:
                    prop_val = props[i]
                normalized_props.append((prop_val - prop_stats[prop]['mean']) / prop_stats[prop]['std'])
            item['properties'] = torch.tensor(normalized_props, dtype=torch.float32)
            
        return item

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        item = self.data[idx]
        
        # Convert sequence to tensor of indices
        sequence_tensor = torch.tensor([self.alphabet.aa_to_idx[aa] for aa in item['sequence']], 
                                     dtype=torch.long)
        
        # Convert coordinates to numpy array for distance calculation
        coords = np.array(item['coordinates'])
        
        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(coords)
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)
        
        # Initialize the output dictionary
        processed_item = {
            'id': item['id'],
            'sequence': self.pad_sequence(sequence_tensor, self.max_seq_length),
            'distance_matrix': self.pad_matrix(dist_tensor, self.max_seq_length),
            'energy': torch.tensor(item['energy'], dtype=torch.float32),
            'length': item['length'],
            'properties': torch.tensor([
                item['properties']['hydrophobic_ratio'],
                item['properties']['charged_ratio']
            ], dtype=torch.float32)
        }
        
        # Add relative position encoding if requested
        if self.include_relative_encoding:
            pos_enc = self.compute_relative_position_encoding(dist_matrix)
            pos_enc_tensor = torch.tensor(pos_enc, dtype=torch.float32)
            processed_item['position_encoding'] = self.pad_matrix(pos_enc_tensor, self.max_seq_length)

        # Normalize features using dataset statistics
        processed_item = self.normalize_features(processed_item)

        # Cache the processed item
        self._cache[idx] = processed_item
        return processed_item

    @classmethod
    def load(cls, dataset_dir: str, include_relative_encoding: bool = True) -> 'PreprocessedProteinDataset':
        """Load dataset from a directory"""
        with open(os.path.join(dataset_dir, 'data.json'), 'r') as f:
            data = json.load(f)
        with open(os.path.join(dataset_dir, 'alphabet.pkl'), 'rb') as f:
            alphabet = pickle.load(f)
        return cls(data, alphabet, include_relative_encoding, dataset_dir)

    def calculate_dataset_statistics(self) -> Dict:
        """Calculate global statistics for the entire dataset"""
        print("Calculating dataset statistics...")
        
        all_distances = []
        all_hydrophobic_ratios = []
        all_charged_ratios = []
        
        for item in tqdm(self.data):
            # Distance statistics
            coords = np.array(item['coordinates'])
            dist_matrix = self.compute_distance_matrix(coords)
            triu_indices = np.triu_indices(dist_matrix.shape[0], k=1)
            distances = dist_matrix[triu_indices]
            all_distances.extend(distances.tolist())
            
            # Property statistics
            all_hydrophobic_ratios.append(item['properties']['hydrophobic_ratio'])
            all_charged_ratios.append(item['properties']['charged_ratio'])
        
        return {
            'distance_stats': {
                'global_stats': {
                    'mean_distance': float(np.mean(all_distances)),
                    'std_distance': float(np.std(all_distances)),
                    'min_distance': float(np.min(all_distances)),
                    'max_distance': float(np.max(all_distances)),
                }
            },
            'property_stats': {
                'hydrophobic_ratio': {
                    'mean': float(np.mean(all_hydrophobic_ratios)),
                    'std': float(np.std(all_hydrophobic_ratios))
                },
                'charged_ratio': {
                    'mean': float(np.mean(all_charged_ratios)),
                    'std': float(np.std(all_charged_ratios))
                }
            }
        }

    def preprocess_and_save(self, output_dir: str):
        """Preprocess entire dataset and save to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate and save dataset statistics
        dataset_stats = self.calculate_dataset_statistics()
        with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        # Save the raw data and alphabet
        with open(os.path.join(output_dir, 'data.json'), 'w') as f:
            json.dump(self.data, f, indent=2)
        with open(os.path.join(output_dir, 'alphabet.pkl'), 'wb') as f:
            pickle.dump(self.alphabet, f)
        
        # Process and save each protein
        print("Processing and saving individual proteins...")
        for idx in tqdm(range(len(self))):
            processed_item = self.__getitem__(idx)
            torch.save(processed_item, os.path.join(output_dir, f'protein_{idx}.pt'))

def main():
    # Base directory containing the dataset splits
    base_dir = 'protein_dataset'
    output_base_dir = 'preprocessed_protein_dataset'
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} dataset...")
        
        # Load and preprocess the dataset
        input_dir = os.path.join(base_dir, split)
        dataset = PreprocessedProteinDataset.load(input_dir)
        
        # Preprocess and save to disk
        output_dir = os.path.join(output_base_dir, split)
        dataset.preprocess_and_save(output_dir)
        
        print(f"Saved preprocessed {split} dataset to {output_dir}")
        print(f"Number of samples: {len(dataset)}")

    print("\nPreprocessing completed successfully!")
    print(f"Preprocessed datasets saved in '{output_base_dir}' directory")

if __name__ == "__main__":
    main()
