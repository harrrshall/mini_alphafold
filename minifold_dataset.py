import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from tqdm.auto import tqdm
import os
from multiprocessing import Manager
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

@dataclass
class AminoAcid:
    __slots__ = ['code', 'hydrophobicity', 'size', 'charge', 'secondary_structure_preference']
    code: str
    hydrophobicity: float
    size: float
    charge: float
    secondary_structure_preference: str

class ProteinEnvironment:
    __slots__ = ['pH', 'temperature', 'ionic_strength']
    def __init__(self, pH: float = 7.0, temperature: float = 37.0, ionic_strength: float = 0.15):
        self.pH = pH
        self.temperature = temperature
        self.ionic_strength = ionic_strength

class EnhancedAminoAcidAlphabet:
    def __init__(self):
        # Pre-compute amino acid objects
        self.amino_acids = {
            'H': AminoAcid('H', 0.9, 1.0, 0, 'helix'),
            'P': AminoAcid('P', 0.1, 0.8, 0, 'coil'),
            'N': AminoAcid('N', 0.5, 0.9, 0, 'sheet'),
            'C': AminoAcid('C', 0.8, 1.1, 1, 'helix'),
            'A': AminoAcid('A', 0.2, 0.7, -1, 'coil')
        }
        # Pre-compute amino acid pairs for faster lookup
        self.aa_pairs = {}
        for code1, aa1 in self.amino_acids.items():
            for code2, aa2 in self.amino_acids.items():
                self.aa_pairs[(code1, code2)] = (aa1, aa2)

        # Add PyTorch-specific mappings
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids.keys())}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_to_idx.items()}

    def get_amino_acid_pair(self, code1: str, code2: str) -> Tuple[AminoAcid, AminoAcid]:
        return self.aa_pairs[(code1, code2)]

    def sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to tensor of indices"""
        return torch.tensor([self.aa_to_idx[aa] for aa in sequence], dtype=torch.long)

class EnhancedEnergyFunction:
    def __init__(self, environment: ProteinEnvironment):
        self.environment = environment
        self.contact_threshold = 4.0
        self.contact_threshold_squared = 16.0
        self.electrostatic_threshold = 8.0
        self.electrostatic_threshold_squared = 64.0
        
    @staticmethod
    def compute_distances_squared(coords: np.ndarray) -> np.ndarray:
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sum(diff * diff, axis=-1)
        
    def calculate_total_energy(self, sequence: str, coords: np.ndarray, alphabet: EnhancedAminoAcidAlphabet) -> float:
        distances_squared = self.compute_distances_squared(coords)
        total_energy = 0.0
        
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                dist_sq = distances_squared[i, j]
                aa1, aa2 = alphabet.get_amino_acid_pair(sequence[i], sequence[j])
                
                if dist_sq <= self.contact_threshold_squared:
                    total_energy += -1.0 * aa1.hydrophobicity * aa2.hydrophobicity / np.sqrt(dist_sq)
                
                if dist_sq <= self.electrostatic_threshold_squared:
                    total_energy += (aa1.charge * aa2.charge) / (self.environment.ionic_strength * np.sqrt(dist_sq))
                
                min_distance = (aa1.size + aa2.size) * 2
                if dist_sq < min_distance * min_distance:
                    total_energy += 100.0
                
                if aa1.secondary_structure_preference == aa2.secondary_structure_preference:
                    total_energy += -0.5
                    
        return total_energy

class EnhancedProteinFolder:
    def __init__(self, energy_function: EnhancedEnergyFunction, alphabet: EnhancedAminoAcidAlphabet):
        self.energy_function = energy_function
        self.alphabet = alphabet
        
    def initialize_coords(self, sequence_length: int) -> np.ndarray:
        return np.random.randn(sequence_length, 2) * 5
        
    def simulated_annealing(self, sequence: str, 
                           initial_temp: float = 100.0,
                           final_temp: float = 0.1,
                           steps: int = 500) -> Tuple[np.ndarray, float]:
        coords = self.initialize_coords(len(sequence))
        current_energy = self.energy_function.calculate_total_energy(sequence, coords, self.alphabet)
        best_coords = coords.copy()
        best_energy = current_energy
        
        temp_schedule = initial_temp * (final_temp / initial_temp) ** (np.arange(steps) / steps)
        
        for step, temperature in enumerate(temp_schedule):
            idx = random.randrange(len(sequence))
            new_coords = coords.copy()
            new_coords[idx] += np.random.randn(2) * 0.5
            
            new_energy = self.energy_function.calculate_total_energy(sequence, new_coords, self.alphabet)
            
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / temperature):
                coords = new_coords
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_coords = coords.copy()
                    best_energy = current_energy
        
        return best_coords, best_energy

def process_batch(params):
    start_idx, num_sequences, generator, progress_dict, batch_id = params
    batch_results = []
    
    for i in range(num_sequences):
        if random.random() < 0.3:
            sequence = generator.generate_designed_sequence(random.choice(generator.patterns))
        else:
            sequence = generator.generate_random_sequence()
            
        coords, energy = generator.folder.simulated_annealing(sequence)
        
        sample = {
            'id': f'protein_{start_idx + i}',
            'sequence': sequence,
            'coordinates': coords.tolist(),
            'energy': float(energy),
            'length': len(sequence),
            'properties': {
                'hydrophobic_ratio': sequence.count('H') / len(sequence),
                'charged_ratio': (sequence.count('C') + sequence.count('A')) / len(sequence)
            }
        }
        batch_results.append(sample)
        
        progress_dict[batch_id] = i + 1
    
    return batch_results

class ProteinDataset(Dataset):
    """PyTorch Dataset for protein structures"""
    def __init__(self, data: List[Dict], alphabet: EnhancedAminoAcidAlphabet):
        self.data = data
        self.alphabet = alphabet

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert sequence to tensor of indices
        sequence_tensor = self.alphabet.sequence_to_tensor(item['sequence'])
        
        # Convert coordinates to tensor
        coords_tensor = torch.tensor(item['coordinates'], dtype=torch.float32)
        
        # Convert energy to tensor
        energy_tensor = torch.tensor(item['energy'], dtype=torch.float32)
        
        # Convert properties to tensor
        properties_tensor = torch.tensor([
            item['properties']['hydrophobic_ratio'],
            item['properties']['charged_ratio']
        ], dtype=torch.float32)
        
        return {
            'id': item['id'],
            'sequence': sequence_tensor,
            'coordinates': coords_tensor,
            'energy': energy_tensor,
            'length': item['length'],
            'properties': properties_tensor
        }

class ToyDatasetGenerator:
    def __init__(self, 
                 alphabet: EnhancedAminoAcidAlphabet,
                 folder: EnhancedProteinFolder,
                 min_length: int = 20,
                 max_length: int = 50):
        self.alphabet = alphabet
        self.folder = folder
        self.min_length = min_length
        self.max_length = max_length
        self.patterns = ['HP', 'HPN', 'HPNCA', 'HHPP']
        
    def generate_random_sequence(self) -> str:
        length = random.randint(self.min_length, self.max_length)
        return ''.join(random.choice(list(self.alphabet.amino_acids.keys())) for _ in range(length))
        
    def generate_designed_sequence(self, pattern: str) -> str:
        length = random.randint(self.min_length, self.max_length)
        return ''.join(pattern[i % len(pattern)] for i in range(length))
        
    def generate_dataset(self, num_samples: int, include_patterns: bool = True) -> List[Dict]:
        num_cores = multiprocessing.cpu_count()
        batch_size = max(1, num_samples // num_cores)
        batches = []
        
        with Manager() as manager:
            progress_dict = manager.dict()
            
            for i in range(0, num_samples, batch_size):
                batch_num_sequences = min(batch_size, num_samples - i)
                batch_id = i // batch_size
                batches.append((i, batch_num_sequences, self, progress_dict, batch_id))
            
            print(f"\nGenerating dataset with {num_cores} processes")
            print(f"Total samples: {num_samples}")
            print(f"Batch size: {batch_size}")
            print(f"Number of batches: {len(batches)}\n")
            
            with tqdm(total=num_samples, desc="Total Progress", position=0) as pbar:
                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    futures = [executor.submit(process_batch, batch) for batch in batches]
                    
                    completed = 0
                    while completed < num_samples:
                        current_completed = sum(progress_dict.values())
                        if current_completed > completed:
                            pbar.update(current_completed - completed)
                            completed = current_completed
                    
                    results = [future.result() for future in futures]
        
        dataset = [item for batch in results for item in batch]
        return dataset

class ProteinDataset(Dataset):
    """PyTorch Dataset for protein structures with persistence capabilities"""
    def __init__(self, data: List[Dict], alphabet: EnhancedAminoAcidAlphabet):
        self.data = data
        self.alphabet = alphabet
        self._cache = {}  # Memory cache for frequently accessed items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        item = self.data[idx]
        
        # Convert sequence to tensor of indices
        sequence_tensor = self.alphabet.sequence_to_tensor(item['sequence'])
        
        # Convert coordinates to tensor
        coords_tensor = torch.tensor(item['coordinates'], dtype=torch.float32)
        
        # Convert energy to tensor
        energy_tensor = torch.tensor(item['energy'], dtype=torch.float32)
        
        # Convert properties to tensor
        properties_tensor = torch.tensor([
            item['properties']['hydrophobic_ratio'],
            item['properties']['charged_ratio']
        ], dtype=torch.float32)
        
        processed_item = {
            'id': item['id'],
            'sequence': sequence_tensor,
            'coordinates': coords_tensor,
            'energy': energy_tensor,
            'length': item['length'],
            'properties': properties_tensor
        }

        # Cache the processed item
        self._cache[idx] = processed_item
        return processed_item

    @classmethod
    def load(cls, dataset_dir: str) -> 'ProteinDataset':
        """Load dataset from a directory"""
        with open(os.path.join(dataset_dir, 'data.json'), 'r') as f:
            data = json.load(f)
        with open(os.path.join(dataset_dir, 'alphabet.pkl'), 'rb') as f:
            alphabet = pickle.load(f)
        return cls(data, alphabet)

    def save(self, dataset_dir: str):
        """Save dataset to a directory"""
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the raw data
        with open(os.path.join(dataset_dir, 'data.json'), 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Save the alphabet
        with open(os.path.join(dataset_dir, 'alphabet.pkl'), 'wb') as f:
            pickle.dump(self.alphabet, f)
        
        # Save dataset metadata
        metadata = {
            'size': len(self),
            'amino_acids': list(self.alphabet.amino_acids.keys()),
            'min_length': min(item['length'] for item in self.data),
            'max_length': max(item['length'] for item in self.data),
            'mean_energy': sum(item['energy'] for item in self.data) / len(self.data),
            'dataset_stats': self._calculate_dataset_stats()
        }
        
        with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def _calculate_dataset_stats(self) -> Dict:
        """Calculate useful statistics about the dataset"""
        all_energies = [item['energy'] for item in self.data]
        all_lengths = [item['length'] for item in self.data]
        
        return {
            'energy_stats': {
                'mean': np.mean(all_energies),
                'std': np.std(all_energies),
                'min': min(all_energies),
                'max': max(all_energies)
            },
            'length_stats': {
                'mean': np.mean(all_lengths),
                'std': np.std(all_lengths),
                'min': min(all_lengths),
                'max': max(all_lengths)
            },
            'amino_acid_distribution': self._get_aa_distribution()
        }

    def _get_aa_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of amino acids in the dataset"""
        aa_count = {aa: 0 for aa in self.alphabet.amino_acids.keys()}
        total_aa = 0
        
        for item in self.data:
            for aa in item['sequence']:
                aa_count[aa] += 1
                total_aa += 1
        
        return {aa: count/total_aa for aa, count in aa_count.items()}

class DatasetSplitter:
    """Helper class to split dataset into train/val/test sets"""
    @staticmethod
    def split_dataset(dataset: ProteinDataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                     random_seed=42) -> Tuple[ProteinDataset, ProteinDataset, ProteinDataset]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
        
        random.seed(random_seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return (
            ProteinDataset([dataset.data[i] for i in train_indices], dataset.alphabet),
            ProteinDataset([dataset.data[i] for i in val_indices], dataset.alphabet),
            ProteinDataset([dataset.data[i] for i in test_indices], dataset.alphabet)
        )

def main():
    print("Initializing protein folding simulation...")
    
    # Create output directory
    dataset_dir = 'protein_dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    
    environment = ProteinEnvironment(pH=7.0, temperature=37.0, ionic_strength=0.15)
    alphabet = EnhancedAminoAcidAlphabet()
    energy_function = EnhancedEnergyFunction(environment)
    folder = EnhancedProteinFolder(energy_function, alphabet)
    generator = ToyDatasetGenerator(alphabet, folder)
    
    print("\nStarting dataset generation...")
    raw_dataset = generator.generate_dataset(num_samples=5000, include_patterns=True)
    
    # Create PyTorch dataset
    full_dataset = ProteinDataset(raw_dataset, alphabet)
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(full_dataset)
    
    # Save datasets
    print("\nSaving datasets...")
    train_dataset.save(os.path.join(dataset_dir, 'train'))
    val_dataset.save(os.path.join(dataset_dir, 'val'))
    test_dataset.save(os.path.join(dataset_dir, 'test'))
    
    # Create example of how to load and use the dataset
    usage_example = """
# Example usage in future sessions:
from torch.utils.data import DataLoader

# Load the datasets
train_dataset = ProteinDataset.load('protein_dataset/train')
val_dataset = ProteinDataset.load('protein_dataset/val')
test_dataset = ProteinDataset.load('protein_dataset/test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""
    
    with open(os.path.join(dataset_dir, 'usage_example.py'), 'w') as f:
        f.write(usage_example)
    
    print("\nDataset generation and organization completed successfully!")
    print(f"Generated {len(raw_dataset)} total protein structures")
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"\nDatasets saved in '{dataset_dir}' directory")
    print("See 'usage_example.py' for loading instructions")

if __name__ == "__main__":
    main()
