from __future__ import print_function
import sys
import yaml
import torch
import collections
import os
from read_data import read_data
from fedn.utils.helpers.helpers import get_helper, save_metadata
from models import pytorch_model
import time
from tqdm import tqdm
# FEDn helper modülünü tanımlıyoruz
HELPER_MODULE = "numpyhelper"

def weights_to_np(weights):
    """Converts PyTorch model weights to numpy arrays for FEDn helper."""
    weights_np = []
    for w in weights:
        weights_np.append(weights[w].cpu().detach().numpy())
    return weights_np

def np_to_weights(weights_np, model):
    """Converts numpy arrays from FEDn helper to PyTorch model weights."""
    state_dict = collections.OrderedDict()
    model_state = model.state_dict()
    
    # Map numpy arrays to PyTorch model parameters
    for i, key in enumerate(model_state.keys()):
        # Make sure we have enough weights
        if i < len(weights_np):
            state_dict[key] = torch.tensor(weights_np[i])
        else:
            raise ValueError(f"Model parameter count mismatch: numpy model has {len(weights_np)} arrays, PyTorch model expects more.")
    
    return state_dict



def train(model, data_path, settings):
    """Train loop with per-epoch logging."""
    
    print("-- RUNNING TRAINING --", flush=True)
    trainset = read_data(data_path)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=settings.get('batch_size', 32),
        shuffle=True,
        num_workers=settings.get('num_workers', 4),
        pin_memory=False)

    # Log ayarları
    total_epochs   = settings.get('epochs', 1)
    log_frequency  = settings.get('log_frequency', 100)   # her 100 batch'te bir ara log
    device         = next(model.parameters()).device

    global_start = time.time()
    num_examples  = 0

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        batch_losses = []           # Opsiyonel: ileride std/median bakmak için
        loader = tqdm(train_loader,
                      desc=f"Epoch {epoch}/{total_epochs}",
                      unit="batch",
                      leave=False)  # leave=True isterseniz çubuklar ekranda kalır

        for step, (x, y) in enumerate(loader, 1):

            batch_size = x.size(0)
            x = x.squeeze(1).float().to(device)
            y = y.to(device)

            # ---------- forward ----------
            model.fit(x, y)

            # ---------- loss / backward ----------
            output = model(x)
            loss_val = -1 * (y * torch.log(output) + (1 - y) * torch.log(1 - output))

            # ---------- logging ----------
            running_loss += loss_val.item()
            batch_losses.append(loss_val.item())
            num_examples += batch_size

            if step % log_frequency == 0:
                loader.set_postfix(loss=f"{loss_val.item():.4f}")

        # ----- epoch summary -----
        epoch_time = time.time() - epoch_start
        avg_loss   = running_loss / step
        print(f"[Epoch {epoch}/{total_epochs}] "
              f"avg_loss={avg_loss:.4f} "
              f"time={epoch_time:.1f}s "
              f"throughput={int(num_examples/epoch_time)} samples/s",
              flush=True)

    total_time = time.time() - global_start
    print(f"-- TRAINING COMPLETED in {total_time/60:.1f} min --", flush=True)
    return model, num_examples


if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python train.py <input_model> <output_model> [data_path]", flush=True)
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    
    # Determine data path:
    # 1. First check environment variable
    # 2. Use argument if provided
    # 3. Try default paths if neither exists
    data_path = os.environ.get('FEDN_TRAIN_CSV')
    if not data_path and len(sys.argv) > 3:
        data_path = sys.argv[3]
    
    # If no path specified or file doesn't exist, try alternative paths
    if not data_path or not os.path.exists(data_path):
        if data_path:
            print(f"WARNING: Specified file not found: {data_path}", flush=True)
        
        if not data_path or not os.path.exists(data_path):
            print(f"WARNING: No data file found. Please specify the correct path to CSV file.", flush=True)
            print(f"Either set the FEDN_TRAIN_CSV environment variable or provide the CSV path as a command line parameter.", flush=True)
            sys.exit(1)
    
    print(f"Input model: {input_model}", flush=True)
    print(f"Output model: {output_model}", flush=True)
    print(f"Data path: {data_path}", flush=True)

    # FEDn'nin helper sistemini kullanıyoruz
    helper = get_helper(HELPER_MODULE)
    
    model = pytorch_model.create_seed_model()
    print('Model created', flush=True)
    
    # Load model
    try:
        model_weights = helper.load(input_model)
        model.load_state_dict(np_to_weights(model_weights, model))
        print("Model successfully loaded", flush=True)
    except Exception as e:
        print(f"Model loading error: {e}", flush=True)
        print("WARNING: Model could not be loaded, continuing with a new model", flush=True)
    
    # Training process
    settings={}
    settings["batch_size"]=32
    settings["epochs"]=5
    model, num_examples_trained = train(model, data_path, settings)
    
    # Save model
    try:
        helper.save(weights_to_np(model.state_dict()), output_model)
        print(f"Model saved: {output_model}", flush=True)
        
        # Metadata oluştur ve kaydet
        metadata = {
            "num_examples": num_examples_trained,
            "batch_size": settings['batch_size'],
            "epochs": settings['epochs'],
            "model_type": "pytorch"
        }
        save_metadata(metadata, output_model)
        print(f"Model metadata saved to: {output_model}-metadata", flush=True)
    except Exception as e:
        print(f"Model saving error: {e}", flush=True)
        raise