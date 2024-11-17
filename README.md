# Mini AlphaFold: A Toy Model for Protein Folding 🧬

This repository contains a simplified implementation of AlphaFold, along with a toy dataset, for educational purposes.  It allows users to experiment with protein folding concepts and simplified machine learning models.

## Dataset Generation and Preprocessing 🧪

The dataset is generated and preprocessed using the following scripts:

* **`minifold_dataset.py`**:  Generates a toy protein dataset with a reduced amino acid alphabet and a simplified energy function. The generated data is split into train, validation, and test sets and saved in the `protein_dataset` directory. 📂
* **`preprocess_dataset.py`**: Preprocesses the generated dataset for use in machine learning models. This involves padding, relative position encoding, normalization, and caching. The preprocessed data is saved in the `preprocessed_protein_dataset` directory. 📂

Refer to the detailed documentation in the previous response for more information on the dataset generation and preprocessing steps.

## Usage Tutorial 📖

This tutorial demonstrates how to use the provided scripts to generate, preprocess, and load the dataset for training or inference.

### 1. Setting up the Environment 🛠️

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   ```

   Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

   Install the required packages (ensure you have a requirements.txt file listing the dependencies like NumPy, SciPy, Matplotlib, tqdm, and PyTorch):

   ```bash
   pip install -r requirements.txt
   ```
2. Generating the Dataset ⚙️

Run the following command to generate the toy protein dataset:

```bash
python minifold_dataset.py
```

This will create the `protein_dataset` directory containing the train, validation, and test sets in JSON and pickle format. It also generates `usage_example.py` inside the `protein_dataset` directory.

3. Preprocessing the Dataset 🔄

Run the following command to preprocess the generated data:

```bash
python preprocess_dataset.py
```

This will create the `preprocessed_protein_dataset` directory containing the preprocessed train, validation, and test sets. Each protein is saved as a separate `.pt` file (PyTorch tensor). Dataset statistics are also saved in this directory.

4. Loading and Using the Dataset 📥

The following code snippet demonstrates how to load and use the preprocessed dataset in PyTorch:

```python
import torch
from torch.utils.data import DataLoader
from preprocessed_protein_dataset import PreprocessedProteinDataset

# Load the preprocessed datasets
train_dataset = PreprocessedProteinDataset.load('preprocessed_protein_dataset/train')
val_dataset = PreprocessedProteinDataset.load('preprocessed_protein_dataset/val')
test_dataset = PreprocessedProteinDataset.load('preprocessed_protein_dataset/test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Iterate through the train loader (example)
for batch in train_loader:
    sequence = batch['sequence']  # Tensor of padded sequence indices
    coords = batch['coordinates'] # Tensor of 2D coordinates
    distance_matrix = batch['distance_matrix'] # Tensor of padded distance matrix
    position_encoding = batch['position_encoding'] # Tensor of padded position encoding
    energy = batch['energy']  # Tensor of protein energies
    length = batch['length'] # Original sequence lengths
    properties = batch['properties']  # Tensor of hydrophobic and charged ratios
    
    # Process the batch data (e.g., feed to your model)
    print(f"Sequence shape: {sequence.shape}")
    print(f"Coordinates shape: {coords.shape}")
    # ... process other features ...
```


## Work in Progress 🚧

This project is currently a work in progress. The core functionality for generating and preprocessing a toy protein dataset is complete. However, the machine learning model and training components are still under development.

### Roadmap 🗺️

The next steps for this project include:

* **`mini_alphafold.py`**: Implement a simplified AlphaFold model architecture. This will involve defining the neural network layers and operations required to predict protein structures from the input features.  **(Empty file created)** 📜
* **`train.py`**:  Develop a training script to train the `mini_alphafold.py` model using the preprocessed dataset. This script will handle training loops, loss calculation, optimization, and model checkpointing. **(Empty file created)** 📜
* **`inference.py`**: Create an inference script to use the trained model for predicting the 3D structure of new protein sequences. This script will load a trained model, preprocess the input sequence, and generate the predicted structure. **(Empty file created)** 📜
* **Model Refinement**: Explore and implement potential improvements to the simplified AlphaFold model, such as incorporating attention mechanisms or more advanced loss functions. ✨
* **Dataset Expansion**: Consider expanding the complexity of the toy dataset by increasing the amino acid alphabet size or incorporating more realistic energy functions. ➕
* **Evaluation Metrics**: Define and implement appropriate evaluation metrics for assessing the performance of the protein folding model. 📊


## Repository Structure 📂

* `minifold_dataset.py`:  Generates the toy protein dataset.
* `preprocess_dataset.py`: Preprocesses the generated dataset.
* `protein_dataset/`: Contains the generated dataset.
* `preprocessed_protein_dataset/`: Contains the preprocessed dataset.
* `mini_alphafold.py`:  **(Empty file - Future implementation of the simplified AlphaFold model)**
* `train.py`: **(Empty file - Future implementation of the training script)**
* `inference.py`: **(Empty file - Future implementation of the inference script)**


This project is intended for educational purposes and provides a simplified environment for learning about protein folding and the associated machine learning techniques.  Contributions and suggestions for improvements are welcome! 👍
