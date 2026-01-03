# Federated Learning for IoT Malware Detection

This repository contains implementation of federated learning for IoT intrusion detection using amino acid encoding feature transformation.

## Project Overview

This project implements a comprehensive federated learning framework for IoT malware detection that addresses the following experimental scenarios defined by supervisors:

1. **Centralized Training (Baseline)** - Training all data on a single node
2. **Federated Learning with Equal Distribution (IID)** - Distributed training with balanced class ratios across all clients
3. **Federated Learning with Skewed Distribution (non-IID)** - Testing robustness under heterogeneous data distributions

Each scenario is tested with and without amino acid encoding to evaluate the contribution of feature engineering.

## Key Features

- **Stratified Data Partitioning**: Each client maintains the exact 60:40 (Malicious:Benign) class distribution as the global dataset
- **Amino Acid Encoding**: Transform numerical network features into protein-like sequences to extract structural properties
- **Multiple Aggregation Strategies**: FedAvg, FedProx, and SCAFFOLD algorithms
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, MCC, FPR
- **FEDn Framework Support**: Optional deployment on physical distributed nodes

## Files Description

### iot_federated_learning_with_amino_acid_encoding.py

**Main experiment script** - Contains all core functionality for running federated learning experiments.

**Key Classes:**
- `ExperimentConfig` - Configuration dataclass for all experiment parameters
- `AminoAcidEncoder` - Converts numerical values to amino acid sequences and extracts structural properties
- `DataPartitioner` - Implements stratified partitioning for IID and non-IID distributions
- `FederatedNetwork` - Neural network model (MLP with BatchNorm, ReLU, Dropout)
- `FederatedClient` - Client node that performs local training
- `FederatedServer` - Central server that coordinates training and aggregates weights
- `CentralizedTrainer` - Baseline centralized training implementation
- `IoTIntrusionDataset` - Dataset loader following team preprocessing methodology
- `ExperimentRunner` - Orchestrates all experiments and generates reports

**Key Methods:**
- `_partition_equal_iid()` - Ensures each client receives data with exact global class ratio (60:40)
- `_partition_skewed_non_iid()` - Creates heterogeneous distributions using Dirichlet distribution
- `train_federated()` - Executes federated learning with configurable number of rounds
- `run_all_experiments()` - Runs all 6 experiment configurations automatically

### fedn_client_with_amino_acid_encoding.py

**FEDn-compatible client** - Standalone implementation for FEDn framework deployment on physical nodes.

**Key Classes:**
- `FednCompatibleClient` - FEDn-compatible client with full training and evaluation capabilities
- `FednAggregator` - Model aggregator supporting multiple strategies (FedAvg, FedProx, SCAFFOLD)
- `ExperimentLogger` - Comprehensive logging and results saving

**Key Methods:**
- `train()` - Train model for specified epochs with verbose output
- `train_local()` - Simple local training matching main script interface
- `get_model_weights()` / `set_model_weights()` - Weight serialization for FEDn communication
- `evaluate()` - Comprehensive evaluation with all metrics

**Note:** This file is designed for FEDn framework deployment. For simulation mode, use the main script directly.

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn torch matplotlib biopython
```

### Running Experiments (Simulation Mode)

```bash
# Run all experiments
python iot_federated_learning_with_amino_acid_encoding.py
```

### Testing FEDn Client

```bash
# Test FEDn-compatible client standalone
python fedn_client_with_amino_acid_encoding.py
```

## Experiment Configuration

Edit the `ExperimentConfig` class to customize experiments:

```python
config = ExperimentConfig(
    # Dataset settings
    dataset_path="./dataset/Malware-Detection-Network-Traffic",
    test_size=0.2,
    
    # Federated learning settings
    num_clients=5,
    num_rounds=50,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    
    # Distribution type: CENTRALIZED, EQUAL_IID, SKEWED_NON_IID
    distribution_type=DistributionType.EQUAL_IID,
    
    # Encoding type: NONE, AMINO_ACID
    encoding_type=EncodingType.AMINO_ACID,
    
    # Model architecture
    input_dim=10,  # Structural properties dimension
    hidden_dims=[100, 50],
    output_dim=2,
    
    # Device
    device="cuda"  # or "cpu"
)
```

## Data Requirements

Place your IoT malware dataset in the specified folder. Expected structure:

```
dataset/
├── capture_1.csv
├── capture_2.csv
└── ...
```

**Required columns** (auto-renamed by script):
- Network flow features (duration, bytes, packets, ports, IPs)
- `label` column: "Malicious" or "Benign"
- `detailed-label` column: Attack type for non-IID partitioning

**Note:** The script expects CSV files with pipe delimiter (|).

## Output Files

After running experiments, the following files are generated:

```
results/
├── experiment_report_YYYYMMDD_HHMMSS.csv  # Summary table
├── full_results_YYYYMMDD_HHMMSS.json       # Complete results
├── experiment_results_YYYYMMDD_HHMMSS.png  # Visualization
└── training_curves_YYYYMMDD_HHMMSS.png     # FL training curves

logs/
└── experiment_YYYYMMDD_HHMMSS.log          # Detailed logs
```

## Running All Experiments

The script automatically runs 6 experiments:

1. `centralized_no_encoding` - Baseline centralized without encoding
2. `centralized_amino_acid` - Centralized with amino acid encoding
3. `fed_equal_iid_no_encoding` - Federated IID without encoding
4. `fed_equal_iid_amino_acid` - Federated IID with encoding
5. `fed_skewed_non_iid_no_encoding` - Federated non-IID without encoding
6. `fed_skewed_non_iid_amino_acid` - Federated non-IID with encoding

## Stratified Partitioning Details

The stratified partitioning ensures each client maintains the exact 60:40 malicious-benign ratio:

```
Global Distribution: 60% Malicious, 40% Benign

Client 0: 60% Malicious + 40% Benign (samples: ~X)
Client 1: 60% Malicious + 40% Benign (samples: ~X)
...
Client N: 60% Malicious + 40% Benign (samples: ~X)
```

This is verified by logging the distribution for each client during partitioning.

## Amino Acid Encoding Details

The encoding scheme maps ASCII characters to amino acids via base-20 (vigesemal) representation:

```
Vigesemal Digits: 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J
Amino Acids:      A R N D C E Q G H I L K M F P S T W Y V
```

This creates a unique amino acid sequence for each feature value, which can then be analyzed for structural properties.

### Structural Properties Extracted

After encoding to amino acid sequences, the following properties are computed:
- Molecular Weight
- Aromaticity
- Instability Index
- Isoelectric Point
- Alpha Helix Fraction
- Reduced Cysteines
- Disulfide Bridges
- Gravy (Grand Average of Hydropathicity)
- Beta Turn Fraction
- Beta Strand Fraction

## FEDn Framework Deployment

To deploy on physical nodes using FEDn:

1. Install FEDn: `pip install fedn`
2. Configure FEDn deployment (see FEDn documentation)
3. Use `FednCompatibleClient` and `FednAggregator` from `fedn_client_with_amino_acid_encoding.py`
4. The client will connect to FEDn aggregator and participate in federated rounds

**Why FEDn?**
- Deploy on actual physical nodes instead of simulated workers
- Better real-world validation
- Industry-relevant contribution for paper
- Network latency and node failure scenarios become real factors

## Performance Metrics

All experiments report:
- **Accuracy** - Overall classification performance
- **Precision** - Weighted precision score
- **Recall** - Weighted recall score
- **F1-Score** - Harmonic mean of precision and recall
- **AUC-ROC** - Area under ROC curve
- **FPR** - False Positive Rate
- **MCC** - Matthews Correlation Coefficient

## Troubleshooting

**Issue: Low accuracy with non-IID data**
- Expected behavior - non-IID distributions are more challenging
- Amino acid encoding should help mitigate this

**Issue: Training takes too long**
- Reduce `num_rounds` for testing (e.g., 20 instead of 50)
- Reduce `num_clients` (e.g., 3 instead of 5)

**Issue: Memory errors**
- Reduce `batch_size`
- Use CPU instead of CUDA if GPU memory is limited

## References

- Original IoT Malware Dataset: CTU-IoT-Malware-Capture
- Federated Learning: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- Amino Acid Encoding: Biopython ProteinAnalysis module

## License

Academic/Research Use

## Authors

- Thaer Al Ibaisi
- Stefan Kuhn
- Muhammad Kazim
