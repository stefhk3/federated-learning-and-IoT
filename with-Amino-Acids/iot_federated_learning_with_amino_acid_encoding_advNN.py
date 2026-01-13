"""
IoT Attack Detection with Federated Learning and Amino Acid Encoding

This module implements a comprehensive federated learning framework for IoT intrusion
detection with DNA/amino acid encoding, covering all experimental cases specified
by supervisors:
1. Centralized training (baseline)
2. Federated learning with equal distribution (IID)
3. Federated learning with skewed distribution (non-IID)
4. All cases with and without amino acid encoding

Authors: Thaer Al Ibaisi, Stefan Kuhn, Muhammad Kazim
Institution: DMU / UT
Version: 4.0.0.0 - Federated Learning Enhanced
"""

import os
import sys
import gc
import time
import logging
import hashlib
import json
import ipaddress
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Data manipulation
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F

# Federated Learning
try:
    import fedn
    FEDN_AVAILABLE = True
except ImportError:
    FEDN_AVAILABLE = False
    print("FEDn not available, using custom federated implementation")

# Amino Acid Encoding
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class EncodingType(Enum):
    """Encoding type enumeration"""
    NONE = "none"
    AMINO_ACID = "amino_acid"
    DNA = "dna"


class DistributionType(Enum):
    """Data distribution type for federated learning"""
    CENTRALIZED = "centralized"
    EQUAL_IID = "equal_iid"  # Equal distribution, IID
    SKEWED_NON_IID = "skewed_non_iid"  # Skewed distribution, non-IID
    ATTACK_CLASS_SPLIT = "attack_class_split"  # Each client gets one attack class


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Dataset settings
    dataset_path: str = "./dataset/Malware-Detection-Network-Traffic"
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Federated learning settings
    num_clients: int = 5
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Data distribution settings
    distribution_type: DistributionType = DistributionType.EQUAL_IID
    skewness_factor: float = 0.9  # For non-IID: how skewed the distribution is
    
    # Encoding settings
    encoding_type: EncodingType = EncodingType.AMINO_ACID
    
    # Model settings
    input_dim: int = 10  # Structural properties dimension
    hidden_dims: List[int] = field(default_factory=lambda: [100, 50])
    output_dim: int = 2
    
    # Logging settings
    log_dir: str = "./logs"
    model_dir: str = "./models"
    results_dir: str = "./results"
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AminoAcidEncoder:
    """
    Implements amino acid encoding for feature transformation.
    
    Maps numerical values to amino acid sequences using ASCII to vigesemal
    conversion followed by amino acid mapping.
    """
    
    # Mapping from vigesemal digits to amino acids
    AMINO_ACID_MAP = {
        '0': 'A',  # Alanine
        '1': 'R',  # Arginine
        '2': 'N',  # Asparagine
        '3': 'D',  # Aspartic acid
        '4': 'C',  # Cysteine
        '5': 'E',  # Glutamic acid
        '6': 'Q',  # Glutamine
        '7': 'G',  # Glycine
        '8': 'H',  # Histidine
        '9': 'I',  # Isoleucine
        'A': 'L',  # Leucine
        'B': 'K',  # Lysine
        'C': 'M',  # Methionine
        'D': 'F',  # Phenylalanine
        'E': 'P',  # Proline
        'F': 'S',  # Serine
        'G': 'T',  # Threonine
        'H': 'W',  # Tryptophan
        'I': 'Y',  # Tyrosine
        'J': 'V',  # Valine
    }
    
    DNA_CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    
    @staticmethod
    def to_vigesemal(decimal_string: str) -> str:
        """Convert decimal string to vigesemal (base-20) representation."""
        dec = int(decimal_string)
        digits = "0123456789ABCDEFGHIJ"
        if dec == 0:
            return "0"
        
        result = ""
        while dec > 0:
            result = digits[dec % 20] + result
            dec //= 20
        return result
    
    @staticmethod
    def map_to_amino(element: str) -> str:
        """Map a vigesemal digit to an amino acid."""
        return AminoAcidEncoder.AMINO_ACID_MAP.get(element, 'A')
    
    @staticmethod
    def convert_to_amino_acid(value: Any) -> str:
        """
        Convert a value to an amino acid sequence.
        
        Args:
            value: Input value (string, number, or convertible to string)
            
        Returns:
            Amino acid sequence string
        """
        try:
            # Convert to string and get ASCII code
            str_value = str(value).strip().upper()
            if not str_value:
                return ""
            
            result = []
            for char in str_value:
                ascii_code = ord(char)
                # Convert to vigesemal (base-20)
                vigesemal = AminoAcidEncoder.to_vigesemal(str(ascii_code))
                # Map each digit to amino acid
                for digit in vigesemal:
                    result.append(AminoAcidEncoder.map_to_amino(digit))
            
            return "".join(result)
        except Exception as e:
            logging.warning(f"Error converting value to amino acid: {e}")
            return ""
    
    @staticmethod
    def encode_dataframe(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Encode entire dataframe using amino acid encoding.
        
        Args:
            df: Input dataframe
            exclude_cols: Columns to exclude from encoding
            
        Returns:
            DataFrame with amino acid encoded features
        """
        exclude_cols = exclude_cols or []
        encoded_df = df.copy()
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            # Apply amino acid encoding to each value
            encoded_df[col] = df[col].apply(AminoAcidEncoder.convert_to_amino_acid)
        
        return encoded_df
    
    @staticmethod
    def sequence_to_features(sequence: str) -> Dict[str, float]:
        """
        Convert amino acid sequence to structural properties.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Dictionary of structural properties
        """
        try:
            if not sequence or len(sequence) < 1:
                return {
                    'MolecularWeight': 0,
                    'Aromaticity': 0,
                    'InstabilityIndex': 0,
                    'IsoelectricPoint': 0,
                    'AlphaHelix': 0,
                    'ReducedCysteines': 0,
                    'DisulfideBridges': 0,
                    'Gravy': 0,
                    'BetaTurn': 0,
                    'BetaStrand': 0
                }
            
            # Use Biopython for structural analysis
            analysed_seq = ProteinAnalysis(sequence)
            
            cysteine_count = analysed_seq.count_amino_acids().get('C', 0)
            return {
                'MolecularWeight': analysed_seq.molecular_weight(),
                'Aromaticity': analysed_seq.aromaticity(),
                'InstabilityIndex': analysed_seq.instability_index(),
                'IsoelectricPoint': analysed_seq.isoelectric_point(),
                'AlphaHelix': analysed_seq.secondary_structure_fraction()[0],
                'ReducedCysteines': cysteine_count,
                'DisulfideBridges': cysteine_count // 2,  # Each disulfide bridge requires 2 cysteines
                'Gravy': analysed_seq.gravy(),
                'BetaTurn': analysed_seq.secondary_structure_fraction()[1],
                'BetaStrand': analysed_seq.secondary_structure_fraction()[2]
            }
        except Exception as e:
            logging.warning(f"Error analyzing sequence: {e}")
            return {
                'MolecularWeight': 0,
                'Aromaticity': 0,
                'InstabilityIndex': 0,
                'IsoelectricPoint': 0,
                'AlphaHelix': 0,
                'ReducedCysteines': 0,
                'DisulfideBridges': 0,
                'Gravy': 0,
                'BetaTurn': 0,
                'BetaStrand': 0
            }
    
    @staticmethod
    def transform_features(df: pd.DataFrame, label_col: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform dataframe to amino acid encoded features.
        
        Args:
            df: Input dataframe with features and label
            label_col: Name of the label column
            
        Returns:
            Tuple of (features, labels)
        """
        # Encode all feature columns
        encoded_df = AminoAcidEncoder.encode_dataframe(
            df, 
            exclude_cols=[label_col]
        )
        
        # Convert to sequences and extract features
        feature_list = []
        labels = []
        
        for idx, row in encoded_df.iterrows():
            # Concatenate all feature values into a sequence
            sequence_parts = []
            for col in encoded_df.columns:
                if col != label_col:
                    sequence_parts.append(str(row[col]))
            sequence = "".join(sequence_parts)
            
            # Extract structural properties
            features = AminoAcidEncoder.sequence_to_features(sequence)
            feature_list.append(list(features.values()))
            
            labels.append(row[label_col])
        
        return np.array(feature_list), np.array(labels)


class DataPartitioner:
    """
    Implements various data partitioning strategies for federated learning.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.attack_types = [
            'Benign', 'Backdoor_Malware', 'BrowserHijacking', 'CommandInjection',
            'DDoS', 'DNS_Spoofing', 'DictionaryBruteForce', 'DoS', 'MITM',
            'Mirai', 'Recon', 'SqlInjection', 'Uploading_Attack', 
            'VulnerabilityScan', 'XSS'
        ]
    
    def partition_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        attack_types: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data based on the configured distribution type.
        
        Args:
            X: Feature matrix
            y: Labels
            attack_types: Attack type labels (for non-IID partitioning)
            
        Returns:
            List of (X_client, y_client) tuples for each client
        """
        if self.config.distribution_type == DistributionType.CENTRALIZED:
            return [(X, y)]
        
        elif self.config.distribution_type == DistributionType.EQUAL_IID:
            return self._partition_equal_iid(X, y)
        
        elif self.config.distribution_type == DistributionType.SKEWED_NON_IID:
            return self._partition_skewed_non_iid(X, y, y)
        
        elif self.config.distribution_type == DistributionType.ATTACK_CLASS_SPLIT:
            if attack_types is None:
                logging.warning("Attack types not provided, using equal distribution")
                return self._partition_equal_iid(X, y)
            return self._partition_by_attack_class(X, y, attack_types)
        
        else:
            raise ValueError(f"Unknown distribution type: {self.config.distribution_type}")
    
    def _partition_equal_iid(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data equally with STRATIFIED distribution.
        
        Ensures each client maintains the same class distribution as the global dataset.
        This is critical for the 60:40 malicious/benign ratio specified by supervisors.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            List of (X_client, y_client) tuples for each client with identical class ratios
        """
        num_clients = self.config.num_clients
        n_samples = len(X)
        
        # Get global class distribution
        unique, counts = np.unique(y, return_counts=True)
        global_ratio = dict(zip(unique.tolist(), (counts / n_samples * 100).tolist()))
        
        logging.info(f"Global class distribution: {global_ratio}")
        print(f"  [STRATIFIED PARTITIONING] Global ratio: Malicious={global_ratio.get(1, 0):.1f}%, Benign={global_ratio.get(0, 0):.1f}%")
        
        # Use stratified sampling to maintain class distribution per client
        partitions = []
        
        for client_id in range(num_clients):
            client_indices = []
            
            # Sample each class separately to maintain exact ratio
            for cls in unique:
                cls_indices = np.where(y == cls)[0]
                n_cls_samples = len(cls_indices) // num_clients
                
                # Distribute samples round-robin style for better mixing
                start_idx = (client_id * n_cls_samples) % len(cls_indices)
                if client_id < num_clients - 1:
                    end_idx = ((client_id + 1) * n_cls_samples) % len(cls_indices)
                    if start_idx < end_idx:
                        client_indices.extend(cls_indices[start_idx:end_idx])
                    else:
                        # Handle wrap-around
                        client_indices.extend(cls_indices[start_idx:])
                        client_indices.extend(cls_indices[:end_idx])
                else:
                    # Last client gets remaining samples
                    remaining = set(cls_indices) - set(client_indices)
                    client_indices.extend(list(remaining))
            
            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)
            
            partitions.append((X[client_indices], y[client_indices]))
            
            # Log this client's distribution
            client_unique, client_counts = np.unique(y[client_indices], return_counts=True)
            client_ratio = dict(zip(client_unique.tolist(), (client_counts / len(client_indices) * 100).tolist()))
            logging.info(f"Client {client_id} distribution: {client_ratio}")
            print(f"    Client {client_id}: Malicious={client_ratio.get(1, 0):.1f}%, Benign={client_ratio.get(0, 0):.1f}% (samples: {len(client_indices)})")
            
        return partitions
    
    def _partition_skewed_non_iid(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        attack_types: np.ndarray = None,
        alpha: float = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data with skew (non-IID distribution) using Dirichlet distribution.
        
        This creates realistic data heterogeneity where each client has a skewed
        subset of the overall data distribution. The alpha parameter controls the
        degree of skew: lower alpha = more skewed distribution.
        
        Args:
            X: Feature matrix
            y: Labels
            attack_types: Attack type labels (for non-IID partitioning)
            alpha: Dirichlet concentration parameter (lower = more skewed)
            
        Returns:
            List of (X_client, y_client) tuples with heterogeneous distributions
        """
        if alpha is None:
            alpha = self.config.skewness_factor
        
        num_clients = self.config.num_clients
        n_samples = len(X)
        
        # Get unique classes
        classes = np.unique(y)
        num_classes = len(classes)
        
        logging.info(f"Creating SKEWED partition with alpha={alpha}")
        print(f"  [SKEWED PARTITIONING] Dirichlet alpha={alpha} (lower = more skewed)")
        
        # Create Dirichlet distribution for each class
        partitions = [([], []) for _ in range(num_clients)]
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            # Dirichlet distribution for class allocation
            # Lower alpha = more extreme proportions (more skew)
            proportions = np.random.dirichlet([alpha] * num_clients)
            cls_proportions = (proportions * len(cls_indices)).astype(int)
            
            # Adjust for rounding to ensure all samples are allocated
            diff = len(cls_indices) - cls_proportions.sum()
            cls_proportions[0] += diff
            
            current = 0
            for client_id in range(num_clients):
                n_cls_samples = cls_proportions[client_id]
                if n_cls_samples > 0 and current < len(cls_indices):
                    end = min(current + n_cls_samples, len(cls_indices))
                    client_indices = cls_indices[current:end]
                    partitions[client_id][0].extend(X[client_indices])
                    partitions[client_id][1].extend(y[client_indices])
                    current = end
        
        # Convert to numpy arrays and log client distributions
        result = []
        for client_id, (X_client, y_client) in enumerate(partitions):
            X_arr = np.array(X_client)
            y_arr = np.array(y_client)
            result.append((X_arr, y_arr))
            
            # Log client distribution
            if len(y_arr) > 0:
                unique, counts = np.unique(y_arr, return_counts=True)
                client_ratio = dict(zip(unique.tolist(), (counts / len(y_arr) * 100).tolist()))
                logging.info(f"Client {client_id} SKEWED distribution: {client_ratio}")
                print(f"    Client {client_id}: Malicious={client_ratio.get(1, 0):.1f}%, Benign={client_ratio.get(0, 0):.1f}% (samples: {len(y_arr)})")
        
        return result
    
    def _partition_by_attack_class(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        attack_types: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data by attack class.
        
        Each client receives data primarily from one attack class,
        simulating a realistic non-IID scenario where different IoT
        devices observe different types of attacks.
        """
        num_clients = self.config.num_clients
        
        # Group indices by attack type
        attack_groups = {}
        for i, attack in enumerate(attack_types):
            if attack not in attack_groups:
                attack_groups[attack] = []
            attack_groups[attack].append(i)
        
        # Assign attack types to clients (some clients may get multiple)
        partitions = [([], []) for _ in range(num_clients)]
        client_attacks = list(attack_groups.keys())
        
        for client_id in range(num_clients):
            # Assign 2-3 attack types to each client
            num_attacks = min(2 + (client_id % 2), len(client_attacks))
            attack_indices = client_attacks[:num_attacks]
            
            for attack in attack_indices:
                if attack in attack_groups:
                    indices = attack_groups[attack]
                    partitions[client_id][0].extend(X[indices])
                    partitions[client_id][1].extend(y[indices])
        
        # Convert to numpy arrays
        result = []
        for X_client, y_client in partitions:
            result.append((np.array(X_client), np.array(y_client)))
        
        return result
    
    @staticmethod
    def analyze_partition_distribution(
        partitions: List[Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Analyze and visualize the distribution of data across partitions.
        
        Args:
            partitions: List of (X, y) tuples
            
        Returns:
            DataFrame with distribution statistics
        """
        stats = []
        for i, (X, y) in enumerate(partitions):
            unique, counts = np.unique(y, return_counts=True)
            stats.append({
                'client': i,
                'total_samples': len(y),
                'num_classes': len(unique),
                'class_distribution': dict(zip(unique.tolist(), counts.tolist()))
            })
        
        return pd.DataFrame(stats)


class FederatedNetwork(nn.Module):
    """
    Neural network model for federated learning.
    
    A multi-layer perceptron with configurable architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(FederatedNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights as a dictionary."""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from a dictionary."""
        self.load_state_dict(weights)


class FederatedClient:
    """
    Federated learning client that performs local training.
    """
    
    def __init__(
        self, 
        client_id: int,
        config: ExperimentConfig,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        self.client_id = client_id
        self.config = config
        
        # Prepare data
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        
        # Create data loader
        dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        # Initialize model
        self.model = FederatedNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim
        ).to(config.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_local(self, num_epochs: int = None) -> float:
        """
        Perform local training on client data.
        
        Args:
            num_epochs: Number of local epochs (uses config if None)
            
        Returns:
            Average training loss
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for epoch in range(num_epochs):
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_y)
                n_samples += len(batch_y)
        
        return total_loss / n_samples if n_samples > 0 else 0.0
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return self.model.get_weights()
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from server."""
        self.model.set_weights(weights)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.config.device)
        y_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        
        metrics = {
            'accuracy': accuracy_score(y_test_np, predictions),
            'precision': precision_score(y_test_np, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test_np, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test_np, predictions, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def get_sample_count(self) -> int:
        """Get number of training samples."""
        return len(self.X_train)


class FederatedServer:
    """
    Federated learning server that coordinates training across clients.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize global model
        self.global_model = FederatedNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim
        ).to(config.device)
        
        self.clients: List[FederatedClient] = []
        self.history = {
            'rounds': [],
            'train_losses': [],
            'val_accuracies': [],
            'client_weights': []
        }
    
    def register_client(self, client: FederatedClient):
        """Register a client with the server."""
        self.clients.append(client)
    
    def aggregate_weights(
        self, 
        weights_list: List[Dict[str, torch.Tensor]],
        sample_counts: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model weights using FedAvg.
        
        Args:
            weights_list: List of model weight dictionaries
            sample_counts: Number of samples for each client
            
        Returns:
            Aggregated weights dictionary
        """
        total_samples = sum(sample_counts)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        for key in weights_list[0].keys():
            # Weighted average
            weighted_sum = torch.zeros_like(weights_list[0][key], dtype=torch.float32)
            
            for weights, count in zip(weights_list, sample_counts):
                weight_factor = count / total_samples
                weighted_sum += weights[key].float() * weight_factor
            
            aggregated_weights[key] = weighted_sum
        
        return aggregated_weights
    
    def broadcast_weights(self, weights: Dict[str, torch.Tensor]):
        """Send global model weights to all clients."""
        for client in self.clients:
            client.set_model_weights(weights)
    
    def train_federated(
        self, 
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute federated learning training.
        
        Args:
            X_test: Global test set
            y_test: Global test labels
            verbose: Whether to print progress
            
        Returns:
            Training history and final metrics
        """
        # Initialize
        global_weights = self.global_model.get_weights()
        n_rounds = self.config.num_rounds
        
        best_accuracy = 0.0
        best_weights = None
        
        if verbose:
            print("\n" + "="*60)
            print("FEDERATED LEARNING TRAINING")
            print("="*60)
            print(f"Number of clients: {len(self.clients)}")
            print(f"Number of rounds: {n_rounds}")
            print(f"Local epochs per round: {self.config.local_epochs}")
            print("="*60 + "\n")
        
        for round_num in range(n_rounds):
            round_start = time.time()
            
            # Store client weights and losses
            client_weights = []
            client_losses = []
            client_sample_counts = []
            
            # Local training on each client
            for client in self.clients:
                # Train locally
                loss = client.train_local()
                client_losses.append(loss)
                client_sample_counts.append(client.get_sample_count())
                
                # Get updated weights
                client_weights.append(client.get_model_weights())
            
            # Aggregate weights (FedAvg)
            aggregated_weights = self.aggregate_weights(
                client_weights, 
                client_sample_counts
            )
            
            # Update global model
            self.global_model.set_weights(aggregated_weights)
            self.broadcast_weights(aggregated_weights)
            
            # Evaluate on test set
            avg_loss = np.mean(client_losses)
            
            # Evaluate global model
            self.global_model.eval()
            X_tensor = torch.FloatTensor(X_test).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.global_model(X_tensor)
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.cpu().numpy()
            
            y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
            accuracy = accuracy_score(y_test_np, predictions)
            
            # Record history
            self.history['rounds'].append(round_num)
            self.history['train_losses'].append(avg_loss)
            self.history['val_accuracies'].append(accuracy)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = {k: v.clone() for k, v in aggregated_weights.items()}
            
            round_time = time.time() - round_start
            
            if verbose and (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1:3d}/{n_rounds} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Test Accuracy: {accuracy:.4f} | "
                      f"Time: {round_time:.2f}s")
        
        # Restore best model
        if best_weights is not None:
            self.global_model.set_weights(best_weights)
        
        # Final evaluation
        final_metrics = self.evaluate_global(X_test, y_test)
        
        if verbose:
            print("\n" + "="*60)
            print("FEDERATED LEARNING COMPLETED")
            print("="*60)
            print(f"Best Test Accuracy: {best_accuracy:.4f}")
            print(f"Final Metrics: {final_metrics}")
            print("="*60 + "\n")
        
        return {
            'history': self.history,
            'best_accuracy': best_accuracy,
            'final_metrics': final_metrics,
            'num_clients': len(self.clients)
        }
    
    def evaluate_global(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.config.device)
        y_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = self.global_model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probabilities_np = probabilities.detach().cpu().numpy()
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        y_test_np = y_tensor.numpy()
        
        try:
            auc = roc_auc_score(y_test_np, probabilities_np[:, 1])
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_test_np, predictions)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        try:
            mcc = matthews_corrcoef(y_test_np, predictions)
        except:
            mcc = 0.0
        
        metrics = {
            'accuracy': accuracy_score(y_test_np, predictions),
            'precision': precision_score(y_test_np, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test_np, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test_np, predictions, average='weighted', zero_division=0),
            'auc_roc': auc,
            'fpr': fpr,
            'mcc': mcc,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics


class CentralizedTrainer:
    """
    Centralized training baseline for comparison.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        self.model = FederatedNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim
        ).to(config.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'loss': [], 'accuracy': []}
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model in centralized manner.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Print progress
            
        Returns:
            Training history and final metrics
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        n_epochs = self.config.num_rounds
        
        if verbose:
            print("\n" + "="*60)
            print("CENTRALIZED TRAINING")
            print("="*60)
            print(f"Training samples: {len(X_train)}")
            print(f"Epochs: {n_epochs}")
            print("="*60 + "\n")
        
        best_val_accuracy = 0.0
        best_model_weights = None
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_samples = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_y)
                n_samples += len(batch_y)
            
            avg_loss = epoch_loss / n_samples
            self.history['loss'].append(avg_loss)
            
            # Validation
            val_metrics = self.evaluate(X_val, y_val)
            self.history['accuracy'].append(val_metrics['accuracy'])
            
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Restore best model
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        
        final_metrics = self.evaluate(X_val, y_val)
        
        if verbose:
            print("\n" + "="*60)
            print("CENTRALIZED TRAINING COMPLETED")
            print("="*60)
            print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
            print(f"Final Metrics: {final_metrics}")
            print("="*60 + "\n")
        
        return {
            'history': self.history,
            'best_accuracy': best_val_accuracy,
            'final_metrics': final_metrics
        }
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.config.device)
        y_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probabilities_np = probabilities.detach().cpu().numpy()
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        y_test_np = y_tensor.numpy()
        
        try:
            auc = roc_auc_score(y_test_np, probabilities_np[:, 1])
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_test_np, predictions)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        try:
            mcc = matthews_corrcoef(y_test_np, predictions)
        except:
            mcc = 0.0
        
        return {
            'accuracy': accuracy_score(y_test_np, predictions),
            'precision': precision_score(y_test_np, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test_np, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test_np, predictions, average='weighted', zero_division=0),
            'auc_roc': auc,
            'fpr': fpr,
            'mcc': mcc
        }


class IoTIntrusionDataset:
    """
    Dataset handler for IoT intrusion detection.
    
    Preprocessing follows the methodology from:
    - CTU-IoT-Malware-Capture dataset
    - Team's preprocessing pipeline (as referenced in research)
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(
        self, 
        data_path: str = None,
        has_attack_types: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load and preprocess the dataset following the team's methodology.
        
        Steps:
        1. Merge all CSV files from dataset folder
        2. Rename columns for clarity
        3. Convert numeric columns
        4. Handle missing values
        5. Process timestamp
        6. Clean data and filter labels
        7. Convert IP addresses to integers
        8. Select features (11 features as per team)
        
        Args:
            data_path: Path to the dataset folder or single file
            has_attack_types: Whether to extract attack type information
            
        Returns:
            Tuple of (X, y, attack_types)
        """
        import ipaddress
        
        # If data_path is a directory, merge all CSV files
        if data_path and os.path.isdir(data_path):
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} CSV files. Merging...")
            
            dfs = []
            for csv_file in sorted(csv_files):
                file_path = os.path.join(data_path, csv_file)
                print(f"  Loading: {csv_file}")
                df_temp = pd.read_csv(file_path, delimiter='|')
                dfs.append(df_temp)
            
            df = pd.concat(dfs, ignore_index=True)
            print(f"Total merged samples: {len(df)}")
            
        elif data_path and data_path.endswith('.csv'):
            df = pd.read_csv(data_path, delimiter='|')
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        
        print(f"Original dataset shape: {df.shape}")
        
        # === TEAM'S PREPROCESSING METHODOLOGY ===
        
        # Step 1: Rename columns for clarity (as per team)
        column_mapping = {
            'ts': 'timestamp',
            'uid': 'unique_id',
            'id.orig_h': 'origin_host_ip',
            'id.orig_p': 'origin_host_port',
            'id.resp_h': 'response_host_ip',
            'id.resp_p': 'response_host_port',
            'proto': 'protocol',
            'orig_bytes': 'origin_bytes',
            'resp_bytes': 'response_bytes',
            'conn_state': 'connection_state',
            'local_orig': 'is_local_origin',
            'local_resp': 'is_local_response',
            'orig_pkts': 'origin_packet_count',
            'orig_ip_bytes': 'origin_ip_bytes',
            'resp_pkts': 'response_packet_count',
            'resp_ip_bytes': 'response_ip_bytes',
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Step 2: Convert numeric columns (as per team)
        numeric_cols = ['duration', 'origin_bytes', 'response_bytes', 'missed_bytes',
                        'origin_packet_count', 'origin_ip_bytes', 
                        'response_packet_count', 'response_ip_bytes']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Step 3: Handle missing values (as per team)
        df[['origin_bytes', 'response_bytes']] = df[['origin_bytes', 'response_bytes']].fillna(0).astype(int)
        df[['service', 'history']] = df[['service', 'history']].fillna('unknown')
        
        # Step 4: Process timestamp (as per team)
        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.iloc[1:].reset_index(drop=True)  # Skip first row
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df.drop('timestamp', axis=1, inplace=True)
        
        # Step 5: Clean data (as per team)
        df.replace('-', np.nan, inplace=True)
        
        # Step 6: Filter labels (as per team)
        df = df[df['label'].isin(['Malicious', 'Benign'])]
        df['label'] = df['label'].map({'Malicious': 1, 'Benign': 0})
        
        # Step 7: Convert IP addresses to integers (as per team)
        df['origin_host_ip'] = df['origin_host_ip'].apply(
            lambda ip: int(ipaddress.IPv4Address(ip)) if pd.notnull(ip) else 0
        )
        df['response_host_ip'] = df['response_host_ip'].apply(
            lambda ip: int(ipaddress.IPv4Address(ip)) if pd.notnull(ip) else 0
        )
        
        print(f"After cleaning: {len(df)} samples")
        
        # === FEATURE SELECTION (11 features as per team) ===
        features = [
            'origin_host_port',
            'response_host_port',
            'origin_ip_bytes',
            'response_ip_bytes',
            'duration',
            'origin_bytes',
            'response_bytes',
            'origin_packet_count',
            'response_packet_count',
            'response_host_ip',
            'origin_host_ip',
        ]
        
        # Extract features and labels
        X = df[features].copy()
        y = df['label'].copy()
        
        # Fill any remaining NaN values in features
        X['duration'] = X['duration'].fillna(0)
        X = X.fillna(0)
        
        # Replace infinity values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Convert to numpy
        X = X.values.astype(np.float32)
        y = y.values.astype(np.int64)
        
        # Extract attack types if available (for non-IID partitioning)
        attack_types = None
        if has_attack_types:
            if 'detailed-label' in df.columns:
                attack_types = df['detailed-label'].values
            else:
                # Use label as attack type for binary case
                attack_types = df['label'].values
        
        return X, y, attack_types
    
    def apply_encoding(
        self, 
        X: np.ndarray,
        encoding_type: EncodingType
    ) -> np.ndarray:
        """
        Apply feature encoding to the data.
        
        Args:
            X: Feature matrix
            encoding_type: Type of encoding to apply
            
        Returns:
            Encoded feature matrix
        """
        if encoding_type == EncodingType.NONE:
            return X
        
        elif encoding_type == EncodingType.AMINO_ACID:
            # Reshape for encoding
            df = pd.DataFrame(X)
            features, labels = AminoAcidEncoder.transform_features(df)
            return features
        
        else:
            return X
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = None,
        val_size: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set (from train)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if test_size is None:
            test_size = self.config.test_size
        if val_size is None:
            val_size = self.config.val_size
        
        # First split: train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,
            random_state=self.config.test_size
        )
        
        # Second split: train and validation
        val_proportion = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_proportion,
            stratify=y_train_val,
            random_state=self.config.test_size
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using standard scaling.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Scaled feature arrays
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled


class ExperimentRunner:
    """
    Main experiment runner that orchestrates all experiments.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config.log_dir, self.config.model_dir, self.config.results_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = os.path.join(
            self.config.log_dir,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_single_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        distribution_type: DistributionType,
        encoding_type: EncodingType,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Run a single experiment with specified configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            distribution_type: Type of data distribution
            encoding_type: Type of feature encoding
            experiment_name: Name for this experiment
            
        Returns:
            Experiment results dictionary
        """
        logging.info(f"Starting experiment: {experiment_name}")
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")
        print(f"Distribution: {distribution_type.value}")
        print(f"Encoding: {encoding_type.value}")
        print(f"{'='*60}")
        
        # Update config for this experiment
        self.config.distribution_type = distribution_type
        self.config.encoding_type = encoding_type
        self.config.input_dim = X_train.shape[1]
        
        # Partition data
        partitioner = DataPartitioner(self.config)
        partitions = partitioner.partition_data(X_train, y_train)
        
        if distribution_type == DistributionType.CENTRALIZED:
            # Centralized training
            trainer = CentralizedTrainer(self.config)
            result = trainer.train(X_train, y_train, X_test, y_test)
            result['experiment_name'] = experiment_name
            result['distribution_type'] = distribution_type.value
            result['encoding_type'] = encoding_type.value
            result['num_clients'] = 1
        else:
            # Federated training
            server = FederatedServer(self.config)
            
            # Register clients
            for i, (X_client, y_client) in enumerate(partitions):
                if len(X_client) > 0:
                    client = FederatedClient(
                        client_id=i,
                        config=self.config,
                        X_train=X_client,
                        y_train=y_client
                    )
                    server.register_client(client)
            
            # Train federated
            result = server.train_federated(X_test, y_test)
            result['experiment_name'] = experiment_name
            result['distribution_type'] = distribution_type.value
            result['encoding_type'] = encoding_type.value
        
        logging.info(f"Experiment {experiment_name} completed")
        print(f"Experiment {experiment_name} completed - "
              f"Accuracy: {result['final_metrics']['accuracy']:.4f}")
        
        return result
    
    def run_all_experiments(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        encoding_filter: EncodingType = None,
        start_experiment: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all experiments as specified by supervisors.
        
        Experiments:
        1. Centralized without encoding (baseline)
        2. Centralized with amino acid encoding
        3. Federated (equal/IID) without encoding
        4. Federated (equal/IID) with amino acid encoding
        5. Federated (skewed/non-IID) without encoding
        6. Federated (skewed/non-IID) with amino acid encoding
        """
        
        # Define all experiment configurations
        experiments = [
            # Baseline: Centralized without encoding
            {
                'name': 'centralized_no_encoding',
                'distribution': DistributionType.CENTRALIZED,
                'encoding': EncodingType.NONE
            },
            # Centralized with amino acid encoding
            {
                'name': 'centralized_amino_acid',
                'distribution': DistributionType.CENTRALIZED,
                'encoding': EncodingType.AMINO_ACID
            },
            # Federated equal/IID without encoding
            {
                'name': 'fed_equal_iid_no_encoding',
                'distribution': DistributionType.EQUAL_IID,
                'encoding': EncodingType.NONE
            },
            # Federated equal/IID with amino acid encoding
            {
                'name': 'fed_equal_iid_amino_acid',
                'distribution': DistributionType.EQUAL_IID,
                'encoding': EncodingType.AMINO_ACID
            },
            # Federated skewed/non-IID without encoding
            {
                'name': 'fed_skewed_non_iid_no_encoding',
                'distribution': DistributionType.SKEWED_NON_IID,
                'encoding': EncodingType.NONE
            },
            # Federated skewed/non-IID with amino acid encoding
            {
                'name': 'fed_skewed_non_iid_amino_acid',
                'distribution': DistributionType.SKEWED_NON_IID,
                'encoding': EncodingType.AMINO_ACID
            },
        ]
        
        if encoding_filter is not None:
            experiments = [
                exp for exp in experiments if exp['encoding'] == encoding_filter
            ]

        if start_experiment:
            start_index = next(
                (i for i, exp in enumerate(experiments) if exp['name'] == start_experiment),
                None
            )
            if start_index is not None:
                experiments = experiments[start_index:]
            else:
                logging.warning(
                    "start_experiment '%s' not found for current encoding filter; running all.",
                    start_experiment
                )

        # Run each experiment
        for exp_config in experiments:
            result = self.run_single_experiment(
                X_train, y_train,
                X_test, y_test,
                distribution_type=exp_config['distribution'],
                encoding_type=exp_config['encoding'],
                experiment_name=exp_config['name']
            )
            
            self.results[exp_config['name']] = result
        
        return self.results
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate a summary report of all experiments.
        
        Returns:
            DataFrame with experiment results
        """
        if not self.results:
            logging.warning("No results to report")
            return pd.DataFrame()
        
        # Collect metrics
        report_data = []
        for exp_name, result in self.results.items():
            row = {
                'experiment': exp_name,
                'distribution': result.get('distribution_type', 'N/A'),
                'encoding': result.get('encoding_type', 'N/A'),
                'num_clients': result.get('num_clients', 1),
                'accuracy': result['final_metrics']['accuracy'],
                'precision': result['final_metrics']['precision'],
                'recall': result['final_metrics']['recall'],
                'f1_score': result['final_metrics']['f1'],
                'auc_roc': result['final_metrics']['auc_roc'],
                'fpr': result['final_metrics']['fpr'],
                'mcc': result['final_metrics']['mcc']
            }
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_path = os.path.join(
            self.config.results_dir,
            f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        report_df.to_csv(report_path, index=False)
        
        logging.info(f"Report saved to {report_path}")
        
        return report_df
    
    def plot_results(self):
        """Generate visualization of experiment results."""
        if not self.results:
            logging.warning("No results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        experiments = list(self.results.keys())
        accuracies = [self.results[exp]['final_metrics']['accuracy'] for exp in experiments]
        f1_scores = [self.results[exp]['final_metrics']['f1'] for exp in experiments]
        aucs = [self.results[exp]['final_metrics']['auc_roc'] for exp in experiments]
        mccs = [self.results[exp]['final_metrics']['mcc'] for exp in experiments]
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(experiments, accuracies, color=['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e74c3c', '#c0392b'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Test Accuracy by Experiment')
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: F1 Score comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(experiments, f1_scores, color=['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e74c3c', '#c0392b'])
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score by Experiment')
        ax2.set_xticklabels(experiments, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: AUC-ROC comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(experiments, aucs, color=['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e74c3c', '#c0392b'])
        ax3.set_ylabel('AUC-ROC')
        ax3.set_title('AUC-ROC by Experiment')
        ax3.set_xticklabels(experiments, rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        for bar, auc in zip(bars3, aucs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: MCC comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(experiments, mccs, color=['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e74c3c', '#c0392b'])
        ax4.set_ylabel('MCC')
        ax4.set_title('Matthews Correlation Coefficient by Experiment')
        ax4.set_xticklabels(experiments, rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        for bar, mcc in zip(bars4, mccs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mcc:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(
            self.config.results_dir,
            f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Results plot saved to {fig_path}")
    
    def plot_training_curves(self):
        """Plot training curves for federated learning."""
        if not self.results:
            return
        
        # Filter federated experiments
        fed_experiments = [
            (name, result) for name, result in self.results.items()
            if 'fed_' in name and 'history' in result
        ]
        
        if not fed_experiments:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss curves
        ax1 = axes[0]
        for name, result in fed_experiments:
            history = result['history']
            ax1.plot(history['rounds'], history['train_losses'], label=name)
        
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Average Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy curves
        ax2 = axes[1]
        for name, result in fed_experiments:
            history = result['history']
            ax2.plot(history['rounds'], history['val_accuracies'], label=name)
        
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Test Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(
            self.config.results_dir,
            f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training curves saved to {fig_path}")


def create_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 20,
    n_attack_types: int = 15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Create a synthetic IoT dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_attack_types: Number of attack types
        random_state: Random seed
        
    Returns:
        Tuple of (DataFrame, labels, attack_types)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features) * 10
    
    # Generate labels (60% benign, 40% attack)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Generate attack types
    attack_types_list = [
        'Benign', 'Backdoor_Malware', 'BrowserHijacking', 'CommandInjection',
        'DDoS', 'DNS_Spoofing', 'DictionaryBruteForce', 'DoS', 'MITM',
        'Mirai', 'Recon', 'SqlInjection', 'Uploading_Attack', 
        'VulnerabilityScan', 'XSS'
    ]
    
    attack_types = []
    for label in labels:
        if label == 0:
            attack_types.append('Benign')
        else:
            attack_types.append(np.random.choice(attack_types_list[1:]))
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    df['protocol'] = np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS', 'MQTT'], n_samples)
    df['device_type'] = np.random.choice(['camera', 'thermostat', 'sensor', 'hub', 'printer'], n_samples)
    
    return df, np.array(labels), np.array(attack_types)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='IoT Attack Detection with Federated Learning and Amino Acid Encoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (default behavior)
  python iot_federated_learning_with_amino_acid_encoding.py
  
  # Run only non-encoded experiments
  python iot_federated_learning_with_amino_acid_encoding.py --run-non-encoded-only
        """
    )
    
    parser.add_argument(
        '--run-non-encoded-only',
        action='store_true',
        default=False,
        help='Run only the non-encoded experiments (Centralized, Federated IID, Federated non-IID without amino acid encoding). Skips the encoded experiments.'
    )

    parser.add_argument(
        '--start-encoding',
        type=str,
        choices=['none', 'amino_acid'],
        default=None,
        help='Resume from a specific encoding section (none or amino_acid).'
    )

    parser.add_argument(
        '--start-experiment',
        type=str,
        choices=[
            'centralized_no_encoding',
            'centralized_amino_acid',
            'fed_equal_iid_no_encoding',
            'fed_equal_iid_amino_acid',
            'fed_skewed_non_iid_no_encoding',
            'fed_skewed_non_iid_amino_acid',
        ],
        default=None,
        help='Resume from a specific experiment name (skips earlier experiments in that encoding section).'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: "auto", "cpu", "cuda", or "cuda:<index>"'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run all experiments.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("IoT ATTACK DETECTION WITH FEDERATED LEARNING AND AMINO ACID ENCODING")
    print("="*70)
    print(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display experiment mode
    if args.run_non_encoded_only:
        print("MODE: Running NON-ENCODED experiments only")
    else:
        print("MODE: Running ALL experiments (encoded and non-encoded)")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU.")
            device = "cpu"

    print(f"DEVICE: {device}")
    
    print("="*70 + "\n")
    
    # Initialize configuration
    config = ExperimentConfig(
        num_clients=5,
        num_rounds=50,
        local_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        test_size=0.2,
        val_size=0.1,
        input_dim=10,
        hidden_dims=[64, 32],
        output_dim=2,
        skewness_factor=0.5,
        device=device
    )
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    try:
        # Load dataset following team's methodology
        print("Loading and preprocessing dataset (following team methodology)...")
        dataset = IoTIntrusionDataset(config)
        
        # Load and preprocess data from dataset folder
        dataset_path = config.dataset_path
        X, y, attack_types = dataset.load_and_preprocess(
            data_path=dataset_path,
            has_attack_types=True
        )
        
        print(f"\nDataset loaded:")
        print(f"  Total samples: {len(y)}")
        print(f"  Benign samples: {np.sum(y == 0)}")
        print(f"  Malicious samples: {np.sum(y == 1)}")
        print(f"  Number of features: {X.shape[1]}")
        
        # Apply amino acid encoding to features
        print("\nApplying amino acid encoding to features...")
        logging.info(f"  Features being encoded: {X.shape[1]}")
        
        encoded_features = []
        for i in range(len(X)):
            row = X[i]
            # Convert each feature value to amino acid sequence
            seq_parts = []
            for val in row:
                seq_parts.append(AminoAcidEncoder.convert_to_amino_acid(val))
            sequence = "".join(seq_parts)
            
            # Get structural properties
            props = AminoAcidEncoder.sequence_to_features(sequence)
            encoded_features.append(list(props.values()))
            
            if (i + 1) % 1000 == 0:
                logging.info(f"  Processed {i + 1}/{len(X)} samples...")
        
        X_encoded = np.array(encoded_features, dtype=np.float32)
        
        print(f"\nAmino Acid Encoding completed:")
        logging.info(f"  Original features: {X.shape[1]}")
        logging.info(f"  Encoded features: {X_encoded.shape[1]} (structural properties)")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # For non-encoded experiments, use original preprocessed features
        X_original = X.copy()
        X_original_scaled = scaler.fit_transform(X_original)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
            X_original_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"\nDataset prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Encoded features: {X_train.shape[1]}")
        print(f"  Original features: {X_train_original.shape[1]}")
        # Handle mixed types (float/string) in attack_types
        attack_type_count = len(np.unique([str(x) for x in attack_types])) if attack_types is not None else 0
        print(f"  Attack types: {attack_type_count}")
        
        # Store both encoded and non-encoded versions
        datasets = {
            'amino_acid': (X_train, X_test, y_train, y_test),
            'none': (X_train_original, X_test_original, y_train_original, y_test_original)
        }
        
        # Run experiments
        all_results = {}
        
        # Determine which encoding types to run based on command-line arguments
        encodings_to_run = ['none'] if args.run_non_encoded_only else list(datasets.keys())

        if args.start_encoding:
            if args.start_encoding in encodings_to_run:
                start_index = encodings_to_run.index(args.start_encoding)
                encodings_to_run = encodings_to_run[start_index:]
            else:
                print(
                    f"WARNING: start-encoding '{args.start_encoding}' not in selected encodings; "
                    "ignoring resume encoding."
                )
        
        encoding_map = {
            'none': EncodingType.NONE,
            'amino_acid': EncodingType.AMINO_ACID
        }

        for encoding_name in encodings_to_run:
            X_tr, X_te, y_tr, y_te = datasets[encoding_name]
            
            print(f"\n{'#'*70}")
            print(f"# EXPERIMENTS WITH {encoding_name.upper()} ENCODING")
            print(f"{'#'*70}")
            
            # Reset results for each encoding
            runner.results = {}
            
            # Run experiments for this encoding
            start_experiment = args.start_experiment if encoding_name == args.start_encoding else None
            results = runner.run_all_experiments(
                X_tr, y_tr, X_te, y_te,
                encoding_filter=encoding_map[encoding_name],
                start_experiment=start_experiment
            )
            all_results[encoding_name] = results
        
        # Generate report
        print("\n" + "="*70)
        print("GENERATING EXPERIMENT REPORT")
        print("="*70)
        
        report = runner.generate_report()
        print("\nExperiment Results Summary:")
        print(report.to_string(index=False))
        
        # Save full results
        results_path = os.path.join(
            config.results_dir,
            f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Convert results to serializable format
        serializable_results = {}
        for encoding, results in all_results.items():
            serializable_results[encoding] = {}
            for exp_name, exp_result in results.items():
                serializable_results[encoding][exp_name] = {
                    'experiment_name': exp_result.get('experiment_name'),
                    'distribution_type': exp_result.get('distribution_type'),
                    'encoding_type': exp_result.get('encoding_type'),
                    'num_clients': exp_result.get('num_clients'),
                    'final_metrics': exp_result.get('final_metrics', {})
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nFull results saved to: {results_path}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        runner.plot_results()
        runner.plot_training_curves()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        logging.error(f"Error running experiments: {e}")
        print(f"\nError: {e}")
        raise
    
    return runner.results


if __name__ == "__main__":
    results = main()
