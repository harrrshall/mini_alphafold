# Mini AlphaFold: A Toy Model for Protein Folding 🧬

This repository provides a simplified, educational implementation of AlphaFold, the groundbreaking protein structure prediction model. It's perfect for learning core concepts of protein folding and exploring associated machine learning techniques.  It includes a toy protein dataset and the building blocks for a simplified model.

## Getting Started 🚀

### 1. Setup 🛠️

* **Virtual Environment:** Create and activate a virtual environment (recommended):
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Install Dependencies:** Install required packages (NumPy, SciPy, Matplotlib, tqdm, and PyTorch):
  ```bash
  pip install -r requirements.txt
  ```

### 2. Generate the Toy Dataset 🧪

```bash
python minifold_dataset.py
```

This script generates a toy protein dataset with a reduced amino acid alphabet and a simplified energy function. The data is split into train, validation, and test sets and saved in the `protein_dataset` directory.  A `usage_example.py` file is also generated within this directory.

### 3. Preprocess the Data ⚙️

```bash
python preprocess_dataset.py
```

This script preprocesses the generated dataset, including padding, relative position encoding, normalization, and caching. The preprocessed data is saved in the `preprocessed_protein_dataset` directory (`.pt` files). Dataset statistics are also saved.

### 4. Loading the Preprocessed Data (PyTorch) ⬇️

```python
import torch
from torch.utils.data import DataLoader
from preprocessed_protein_dataset import PreprocessedProteinDataset

# Load datasets
train_dataset = PreprocessedProteinDataset.load('preprocessed_protein_dataset/train')
val_dataset = PreprocessedProteinDataset.load('preprocessed_protein_dataset/val')
test_dataset = PreprocessedProteinDataset.load('preprocessed_protein_dataset/test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example: Iterate through the train loader ♻️
for batch in train_loader:
    sequence = batch['sequence']          # Padded sequence indices
    coords = batch['coordinates']         # 2D coordinates
    distance_matrix = batch['distance_matrix'] # Padded distance matrix
    position_encoding = batch['position_encoding'] # Padded position encoding
    energy = batch['energy']            # Protein energies
    length = batch['length']            # Original sequence lengths
    properties = batch['properties']      # Hydrophobic and charged ratios

    # Process the batch (e.g., feed to your model)
    print(f"Sequence shape: {sequence.shape}")
    print(f"Coordinates shape: {coords.shape}")
    # ... process other features ...
```

## Model Components 📦

* **`mini_alphafold.py`:**  The main model architecture (`SimplifiedAlphaFold`). Integrates the `Pairformer` and `Diffusion` modules to predict protein structures. Accepts sequences, distance matrices, positional encodings, and other properties as input, outputting 3D coordinates. Includes a `training_step` function.

* **`pairformer.py`:** Implements a simplified Pairformer block using Triangle Attention. Focuses on pairwise amino acid interactions, using distances between residues for more informed attention calculations.

* **`simplediffusion.py`:**  A simplified diffusion model.  Generates protein structures by denoising random Gaussian noise conditioned on sequence embeddings and timesteps.

## Project Roadmap 🗺️

This project is under active development!  Dataset generation and preprocessing are complete.  Model training and inference are next.

* **`train.py`:** (Planned) Training script for `mini_alphafold.py`. Handles training loops, loss calculation, optimization, and checkpointing.
* **`inference.py`:** (Planned) Inference script to predict 3D structures of new sequences.
* **Model Refinement:** (Future)  Advanced loss functions, attention mechanisms, and other improvements.
* **Dataset Expansion:** (Future) Larger amino acid alphabet and/or more realistic energy functions.
* **Evaluation Metrics:** (Future) Implement metrics to assess model performance.


## Repository Contents 📂

* **`minifold_dataset.py`:** Dataset generation
* **`preprocess_dataset.py`:** Dataset preprocessing
* **`mini_alphafold.py`:** Simplified AlphaFold model
* **`train.py`:** (Planned) Training script
* **`inference.py`:** (Planned) Inference script
* **`protein_dataset/`:** Raw dataset
* **`preprocessed_protein_dataset/`:** Preprocessed dataset
* **`pairformer.py`:** Pairformer module
* **`simplediffusion.py`:** Diffusion module
* **`requirements.txt`:** Project dependencies


## Contributing 🤝

Contributions and suggestions are welcome!  This educational project is a simplified starting point for understanding protein folding.
