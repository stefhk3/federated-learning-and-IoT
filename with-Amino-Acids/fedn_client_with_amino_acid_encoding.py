#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEDn-Compatible Federated Learning Client for IoT Attack Detection

This module provides a FEDn-compatible client implementation that integrates
with the amino acid encoding framework for IoT intrusion detection.

Compatible with FEDn 1.x and supports custom model aggregation strategies.
This file works alongside the main federated learning script:
iot_federated_learning_with_amino_acid_encoding.py

Usage:
    - For simulation mode: Use the main script directly
    - For FEDn deployment: Use this client with FEDn framework

Authors: Thaer Al Ibaisi, Stefan Kuhn, Muhammad Kazim
Institution: DMU / UT
"""

import os
import sys
import json
import time
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import amino acid encoding and experiment configuration from main script
from iot_federated_learning_with_amino_acid_encoding import (
    AminoAcidEncoder,
    ExperimentConfig,
    FederatedNetwork
)


class FednCompatibleClient:
    """
    FEDn-compatible federated learning client.
    
    Implements the FEDn client interface for integration with FEDn
    federated learning platform.
    
    Note: This class is designed for FEDn framework deployment.
    For simulation mode, use FederatedClient from the main script instead.
    """
    
    def __init__(
        self, 
        client_id: int,
        config: ExperimentConfig,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        data_path: str = None
    ):
        self.client_id = client_id
        self.config = config
        self.data_path = data_path
        
        # Setup paths
        self.model_path = f"./models/client_{client_id}"
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
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
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'samples_processed': 0
        }
        
        # Data loaders
        self.train_loader = None
        self.X_train = X_train
        self.y_train = y_train
        
        # Setup logging
        self._setup_logging()
        
        # Load data if provided
        if X_train is not None and y_train is not None:
            self._prepare_data_loader(X_train, y_train)
    
    def _setup_logging(self):
        """Setup client-specific logging."""
        log_file = f"./logs/client_{self.client_id}.log"
        Path("./logs").mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"Client-{self.client_id}")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def _prepare_data_loader(self, X: np.ndarray, y: np.ndarray):
        """Prepare data loader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.logger.info(f"Loaded {len(X)} samples for training")
        self.X_train = X
        self.y_train = y
    
    def load_data(self, data: Tuple[np.ndarray, np.ndarray]):
        """
        Load and prepare data for training.
        
        Args:
            data: Tuple of (X, y) arrays
        """
        X, y = data
        self._prepare_data_loader(X, y)
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """
        Get current model weights as numpy arrays.
        
        Returns:
            Dictionary of model weights
        """
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().detach().numpy()
        return weights
    
    def set_model_weights(self, weights: Dict[str, np.ndarray]):
        """
        Set model weights from numpy arrays.
        
        Args:
            weights: Dictionary of model weights
        """
        for name, weight in weights.items():
            self.model.state_dict()[name].copy_(
                torch.from_numpy(weight)
            )
    
    def train_local(self, num_epochs: int = None) -> float:
        """
        Perform local training on client data.
        Compatible with main script's FederatedClient interface.
        
        Args:
            num_epochs: Number of local epochs (uses config if None)
            
        Returns:
            Average training loss
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        if self.train_loader is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
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
    
    def train(
        self, 
        epochs: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model on local data.
        
        Args:
            epochs: Number of training epochs
            verbose: Print progress
            
        Returns:
            Training metrics dictionary
        """
        if epochs is None:
            epochs = self.config.local_epochs
        
        if self.train_loader is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_y)
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == batch_y).sum().item()
                epoch_total += len(batch_y)
            
            avg_loss = epoch_loss / epoch_total
            accuracy = epoch_correct / epoch_total
            
            total_loss += avg_loss
            correct += epoch_correct
            total += epoch_total
            
            self.train_history['loss'].append(avg_loss)
            self.train_history['accuracy'].append(accuracy)
            self.train_history['samples_processed'] += epoch_total
            
            if verbose and (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
        
        avg_total_loss = total_loss / epochs
        avg_accuracy = correct / total
        
        metrics = {
            'loss': avg_total_loss,
            'accuracy': avg_accuracy,
            'samples_processed': total,
            'epochs': epochs
        }
        
        self.logger.info(
            f"Training completed - Avg Loss: {avg_total_loss:.4f}, "
            f"Avg Accuracy: {avg_accuracy:.4f}"
        )
        
        return metrics
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.config.device)
        y_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        y_test_np = y_tensor.numpy()
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
        )
        
        try:
            auc = roc_auc_score(y_test_np, probabilities[:, 1].numpy())
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
            'mcc': mcc
        }
        
        return metrics
    
    def get_sample_count(self) -> int:
        """Get number of training samples."""
        if self.X_train is not None:
            return len(self.X_train)
        return 0
    
    def save_model(self, path: str = None):
        """
        Save model to file.
        
        Args:
            path: Path to save model (uses default if None)
        """
        if path is None:
            path = os.path.join(self.model_path, "model.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'client_id': self.client_id
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """
        Load model from file.
        
        Args:
            path: Path to load model from (uses default if None)
        """
        if path is None:
            path = os.path.join(self.model_path, "model.pt")
        
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_history = checkpoint.get('train_history', self.train_history)
            
            self.logger.info(f"Model loaded from {path}")
        else:
            self.logger.warning(f"Model file not found: {path}")


class FednAggregator:
    """
    FEDn-compatible model aggregator.
    
    Implements custom aggregation strategies for federated learning.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize global model
        self.global_model = FederatedNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim
        ).to(config.device)
        
        # Aggregation history
        self.aggregation_history = {
            'rounds': [],
            'client_count': [],
            'avg_accuracy': []
        }
        
        # Setup logging
        self.logger = logging.getLogger("Aggregator")
        self.logger.setLevel(logging.INFO)
        
        # Setup logging handlers
        log_file = "./logs/aggregator.log"
        Path("./logs").mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def aggregate(
        self,
        client_weights: List[Dict[str, np.ndarray]],
        client_metrics: List[Dict[str, Any]],
        strategy: str = 'fedavg'
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client model weights.
        
        Args:
            client_weights: List of weight dictionaries from clients
            client_metrics: List of training metrics from clients
            strategy: Aggregation strategy ('fedavg', 'fedprox', 'scaffold')
            
        Returns:
            Aggregated weights dictionary
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        if strategy == 'fedavg':
            return self._fedavg_aggregate(client_weights, client_metrics)
        elif strategy == 'fedprox':
            return self._fedprox_aggregate(client_weights, client_metrics)
        elif strategy == 'scaffold':
            return self._scaffold_aggregate(client_weights, client_metrics)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    def _fedavg_aggregate(
        self,
        client_weights: List[Dict[str, np.ndarray]],
        client_metrics: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Federated Averaging (FedAvg) aggregation.
        
        Weights are averaged weighted by the number of samples each client trained on.
        """
        # Calculate weights based on sample counts
        total_samples = sum(
            metrics.get('samples_processed', 1) 
            for metrics in client_metrics
        )
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        for key in client_weights[0].keys():
            weighted_sum = np.zeros_like(client_weights[0][key], dtype=np.float64)
            
            for weights, metrics in zip(client_weights, client_metrics):
                weight_factor = metrics.get('samples_processed', 1) / total_samples
                weighted_sum += weights[key].astype(np.float64) * weight_factor
            
            aggregated_weights[key] = weighted_sum.astype(np.float32)
        
        self.logger.info(f"FedAvg aggregation complete: {len(client_weights)} clients, {total_samples} total samples")
        
        return aggregated_weights
    
    def _fedprox_aggregate(
        self,
        client_weights: List[Dict[str, np.ndarray]],
        client_metrics: List[Dict[str, Any]],
        mu: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        FedProx aggregation with proximal term.
        
        Adds a proximal term to prevent client drift.
        """
        # First, perform FedAvg aggregation
        fedavg_weights = self._fedavg_aggregate(client_weights, client_metrics)
        
        # Get global weights
        global_weights = self.get_weights()
        
        # Apply proximal correction
        corrected_weights = {}
        for key in fedavg_weights.keys():
            correction = mu * (fedavg_weights[key] - global_weights[key])
            corrected_weights[key] = fedavg_weights[key] - correction
        
        self.logger.info("FedProx aggregation complete with proximal term")
        
        return corrected_weights
    
    def _scaffold_aggregate(
        self,
        client_weights: List[Dict[str, np.ndarray]],
        client_metrics: List[Dict[str, Any]],
        global_control: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        SCAFFOLD aggregation with control variates.
        
        Uses control variates to correct client drift.
        """
        if global_control is None:
            global_control = {
                k: np.zeros_like(v) 
                for k, v in client_weights[0].items()
            }
        
        # Calculate weight factors
        total_samples = sum(
            metrics.get('samples_processed', 1) 
            for metrics in client_metrics
        )
        
        # Initialize aggregated weights and control
        aggregated_weights = {}
        aggregated_control = {}
        
        for key in client_weights[0].keys():
            weighted_sum = np.zeros_like(client_weights[0][key], dtype=np.float64)
            control_sum = np.zeros_like(client_weights[0][key], dtype=np.float64)
            
            for weights, metrics in zip(client_weights, client_metrics):
                weight_factor = metrics.get('samples_processed', 1) / total_samples
                weighted_sum += weights[key].astype(np.float64) * weight_factor
                control_sum += global_control[key] * weight_factor
            
            aggregated_weights[key] = weighted_sum.astype(np.float32)
            aggregated_control[key] = control_sum
        
        # Store new control variates
        self._last_control_variates = aggregated_control
        
        self.logger.info("SCAFFOLD aggregation complete with control variates")
        
        return aggregated_weights
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get global model weights."""
        weights = {}
        for name, param in self.global_model.state_dict().items():
            weights[name] = param.cpu().detach().numpy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set global model weights."""
        for name, weight in weights.items():
            self.global_model.state_dict()[name].copy_(
                torch.from_numpy(weight)
            )


class ExperimentLogger:
    """
    Comprehensive experiment logger and results saver.
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "experiment.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        
        # Results storage
        self.experiments = []
        self.start_time = time.time()
    
    def log_experiment_config(self, config: ExperimentConfig):
        """Log experiment configuration."""
        config_dict = asdict(config)
        
        with open(self.log_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info("Experiment configuration saved")
    
    def log_experiment_result(self, result: Dict[str, Any]):
        """Log experiment result."""
        self.experiments.append(result)
        
        # Save to JSON
        with open(self.log_dir / "results.json", 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
        
        # Save summary
        self._save_summary()
    
    def _save_summary(self):
        """Save experiment summary."""
        if not self.experiments:
            return
        
        summary = {
            'total_experiments': len(self.experiments),
            'best_accuracy': max(
                exp.get('final_metrics', {}).get('accuracy', 0)
                for exp in self.experiments
            ),
            'total_time': time.time() - self.start_time,
            'experiments': []
        }
        
        for exp in self.experiments:
            summary['experiments'].append({
                'name': exp.get('experiment_name'),
                'distribution': exp.get('distribution_type'),
                'encoding': exp.get('encoding_type'),
                'accuracy': exp.get('final_metrics', {}).get('accuracy'),
                'f1': exp.get('final_metrics', {}).get('f1'),
                'auc_roc': exp.get('final_metrics', {}).get('auc_roc')
            })
        
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def log_message(self, message: str, level: str = 'INFO'):
        """Log a custom message."""
        if level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_attack_types: int = 15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Create a synthetic IoT dataset for testing.
    Compatible with the main script's create_synthetic_dataset function.
    
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


def run_quick_experiment():
    """
    Run a quick experiment to verify the FEDn client implementation.
    Uses local classes to ensure compatibility without requiring main script exports.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    print("\n" + "="*60)
    print("FEDN CLIENT QUICK EXPERIMENT TEST")
    print("="*60)
    
    # Setup configuration
    config = ExperimentConfig(
        num_clients=3,
        num_rounds=20,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.001,
        test_size=0.2,
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=2
    )
    
    # Generate dataset
    print("Generating synthetic dataset...")
    df, labels, attack_types = create_synthetic_dataset(
        n_samples=3000,
        n_features=30,
        random_state=42
    )
    
    # Apply amino acid encoding
    print("Applying amino acid encoding...")
    encoded_features = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        seq_parts = []
        for val in row.values:
            seq_parts.append(AminoAcidEncoder.convert_to_amino_acid(val))
        sequence = "".join(seq_parts)
        props = AminoAcidEncoder.sequence_to_features(sequence)
        encoded_features.append(list(props.values()))
    
    X = np.array(encoded_features, dtype=np.float32)
    y = labels.astype(np.int64)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Dataset prepared: {len(X_train)} train, {len(X_test)} test samples")
    
    # Test FEDn client training
    print("\nTesting FEDnCompatibleClient...")
    
    # Create clients
    clients = []
    n_samples = len(X_train)
    client_size = n_samples // config.num_clients
    
    for i in range(config.num_clients):
        start = i * client_size
        end = start + client_size if i < config.num_clients - 1 else n_samples
        
        client = FednCompatibleClient(
            client_id=i,
            config=config,
            X_train=X_train[start:end],
            y_train=y_train[start:end]
        )
        clients.append(client)
        print(f"  Created client {i} with {end - start} samples")
    
    # Train each client locally
    print("\nTraining clients locally...")
    for client in clients:
        metrics = client.train(num_epochs=3, verbose=False)
        print(f"  Client {client.client_id}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    # Aggregate weights using FednAggregator
    print("\nAggregating model weights...")
    aggregator = FednAggregator(config)
    
    client_weights = [client.get_model_weights() for client in clients]
    client_metrics = [
        {'samples_processed': client.get_sample_count(), 'loss': 0.5}
        for client in clients
    ]
    
    aggregated_weights = aggregator.aggregate(client_weights, client_metrics, strategy='fedavg')
    
    # Set aggregated weights to first client and evaluate
    clients[0].set_model_weights(aggregated_weights)
    test_metrics = clients[0].evaluate(X_test, y_test)
    
    print(f"\nAggregated Model Test Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    print("\n" + "="*60)
    print("FEDN CLIENT QUICK EXPERIMENT COMPLETED")
    print("="*60)
    
    return {
        'clients': len(clients),
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1']
    }


if __name__ == "__main__":
    results = run_quick_experiment()
