import sys
import os
from read_data import read_data
import json
import yaml
import torch
import collections
from fedn.utils.helpers.helpers import get_helper, save_metrics
from models import pytorch_model
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Define FEDn helper module
HELPER_MODULE = "numpyhelper"
print("Validate.py loaded")
def np_to_weights(weights_np, model):
    """Convert weights to model compatible format"""
    state_dict = collections.OrderedDict()
    
    # If weights_np is already a dictionary
    if isinstance(weights_np, dict):
        for key in weights_np:
            state_dict[key] = torch.tensor(weights_np[key])
        return state_dict
    
    # If weights_np is a list, match using current model state_dict
    try:
        model_state = model.state_dict()
        for i, key in enumerate(model_state.keys()):
            if i < len(weights_np):
                state_dict[key] = torch.tensor(weights_np[i])
    except Exception as e:
        print(f"Error converting weights: {e}", flush=True)
        # If failed, return original structure
        return weights_np
    
    return state_dict

def validate(model, data_path, loss, settings):
    print("-- RUNNING VALIDATION --", flush=True)
    print(f"Reading validation data from: {data_path}", flush=True)

    def evaluate(model, loss, dataloader):
        model.eval()
        meanSquaredError = 0
        meanAverageError = 0
        meanStddev = 0
        meanMase = 0
        total_samples = 0
        processed_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                print("Processing batch...", flush=True)
                batch_size = x.shape[0]
                total_samples += batch_size
                x = torch.squeeze(x, 1)
                x_float = torch.from_numpy(x.float().numpy())

                print("Running model forward pass...", flush=True)
                # Check model type and output type for debugging
                print(f"Model type: {type(model)}", flush=True)
                output = model.forward(x_float)
                print(f"Output type: {type(output)}", flush=True)
                    
                print("Calculating metrics...", flush=True)
                try:
                    # Check and fix output tensor dimensions
                    print(f"Output original shape: {output.shape}", flush=True)
                    if output.dim() > 2:  # If output has more than 2 dimensions
                        output = output.squeeze(-1)  # Squeeze last dimension
                        print(f"Output squeezed shape: {output.shape}", flush=True)
                    
                    # Compare input and output tensors
                    print(f"Comparing shapes - Output: {output.shape}, Input: {input.shape}", flush=True)

                    batch_acc = accuracy_score(y, output)
                    batch_f1 = f1_score(y, output)
                    batch_recall = recall_score(y, output)
                    batch_precision = precision_score(y, output)
                    acc += batch_acc * batch_size
                    f1 += batch_f1 * batch_size
                    recall += batch_recall * batch_size
                    precision += batch_precision * batch_size
                    processed_batches += 1
                    print(f"Batch {processed_batches} metrics: Accuracy={batch_acc}, F!={batch_f1}, Recall={batch_recall}, Precision={batch_precision}", flush=True)
                except Exception as e:
                    print(f"Error calculating metrics: {e}", flush=True)
                    print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'unknown'}", flush=True)
                    print(f"Input shape: {input.shape}", flush=True)
                    raise

            if total_samples > 0:
                meanAverageError /= total_samples
                meanSquaredError /= total_samples
                meanStddev /= total_samples
                meanMase /= total_samples
            else:
                print("Warning: No samples processed", flush=True)
                meanAverageError = float('inf')
                meanSquaredError = float('inf')
                meanStddev = float('inf')
                meanMase = float('inf')
        
        return float(meanAverageError), float(meanSquaredError), float(meanStddev), float(meanMase)

    print("Loading test dataset...", flush=True)
    testset = read_data(data_path)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=settings['batch_size'], shuffle=True)

    try:
        print("Starting evaluation...", flush=True)
        test_mae, test_mse, test_stddev, test_mase = evaluate(model, loss, test_loader)
        print(f"MAE: {test_mae:.6f}, MSE: {test_mse:.6f}, StdDev: {test_stddev:.6f}, MASE: {test_mase:.6f}", flush=True)
    except Exception as e:
        import traceback
        print(f"failed to validate the model {e}", flush=True)
        print(f"Full traceback:", flush=True)
        traceback.print_exc()
        test_mae = float('inf')
        test_mse = float('inf')
        test_stddev = float('inf')
        test_mase = float('inf')
    
    print("Preparing validation report...", flush=True)
    # Create a report with scalar values to be displayed in FEDn Studio UI
    report = { 
                "classification_report": 'unevaluated',
                "MAE": test_mae,
                "MSE": test_mse,
                "StdDev": test_stddev,
                "MASE": test_mase,
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':
    print("Checking command line arguments...", flush=True)
    if len(sys.argv) < 3:
        print("Usage: python validate.py <model_path> <output_path> [data_path]", flush=True)
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("Determining data path...", flush=True)
    data_path = os.environ.get('FEDN_TEST_CSV')
    if not data_path and len(sys.argv) > 3:
        data_path = sys.argv[3]
    
    if not data_path or not os.path.exists(data_path):
        if data_path:
            print(f"WARNING: Specified file not found: {data_path}", flush=True)
        
        if not data_path or not os.path.exists(data_path):
            print(f"WARNING: No data file found. Please specify the correct path to CSV file.", flush=True)
            print(f"Either set the FEDN_TEST_CSV environment variable or provide the CSV path as a command line parameter.", flush=True)
            sys.exit(1)
    
    print(f"Model path: {model_path}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    print(f"Data path: {data_path}", flush=True)

    print("Initializing FEDn helper...", flush=True)
    helper = get_helper(HELPER_MODULE)
    
    print("Creating model...", flush=True)
    model = pytorch_model.create_seed_model()
    print('Model created, starting evaluation', flush=True)
    
    print("Loading model weights...", flush=True)
    try:
        model_weights = helper.load(model_path)
        print(f"Loaded weights type: {type(model_weights)}", flush=True)
        if isinstance(model_weights, list):
            print(f"Model weights is a list with {len(model_weights)} elements", flush=True)
        elif isinstance(model_weights, dict):
            print(f"Model weights is a dict with {len(model_weights.keys())} keys", flush=True)
        
        # Pass current model to conversion function
        weights_dict = np_to_weights(model_weights, model)
        model.load_state_dict(weights_dict)
        print("Model successfully loaded", flush=True)
    except Exception as e:
        print(f"Model loading error: {e}", flush=True)
        print("WARNING: Model could not be loaded, evaluation may fail", flush=True)
        # Normal exit is more appropriate for FEDn in this case
        # Because validation cannot be performed if model cannot be loaded
        report = {
            "classification_report": 'failed',
            "MAE": float('inf'), 
            "MSE": float('inf'),
            "MASE": float('inf'),
            "error": str(e)
        }
        # Use FEDn helper function
        save_metrics(report, output_path)
        sys.exit(1)

    print("Running validation...", flush=True)
    report = validate(model, data_path)

    print("Saving evaluation results...", flush=True)
    try:
        # Use FEDn helper function
        save_metrics(report, output_path)
        print(f"Evaluation report saved: {output_path}", flush=True)
    except Exception as e:
        print(f"Report saving error: {e}", flush=True)
        with open(output_path, "w") as fh:
            fh.write(json.dumps(report))
        print(f"Fallback: Saved report using direct JSON write", flush=True)