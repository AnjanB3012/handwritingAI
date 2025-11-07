# =============================================
# train_handwriting_rnn_v2.py - PyTorch Version
# =============================================
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ---------- PARAMETERS ----------
EPOCHS = 1
BATCH_SIZE = 2
LR = 1e-3
MODEL_PATH = "my_model.pt"
START_COORD_MODEL_PATH = "start_coord_model.pt"
FINISHED_PROGRESS_MODEL_PATH = "finished_progress_model.pt"
DATA_FILE = "input_data.json"

# Set device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ---------- DATASET ----------
class HandwritingDataset(Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            self.entries = json.load(f)

        self.samples = []
        all_deltas = []
        self.start_coords = []  # Store starting coordinates for each sample
        self.len_first_word_list = []
        self.len_text_list = []

        # Parse each entry - keep ALL raw points without merging
        for entry in self.entries:
            text = entry["entry_text"]
            stroke_data = entry["stroke_data"]
            # Sort points by timestamp to maintain temporal order
            points = sorted(stroke_data.values(), key=lambda p: p["timestamp"])
            if len(points) < 2:
                continue
            
            # Store starting coordinates (0th point)
            start_x = points[0]["coordinates"][0]
            start_y = points[0]["coordinates"][1]
            self.start_coords.append([start_x, start_y])
            
            t0 = points[0]["timestamp"]

            # Build sequence from ALL points - no merging, maximum precision
            seq = []
            for i in range(1, len(points)):
                prev, curr = points[i - 1], points[i]
                # Compute deltas with full precision
                dx = curr["coordinates"][0] - prev["coordinates"][0]
                dy = curr["coordinates"][1] - prev["coordinates"][1]
                dt = curr["timestamp"] - t0
                # Keep all raw values - pressure and tilt as-is
                seq.append([dx, dy, dt, curr["pressure"], curr["tilt"]])
            
            # Use float32 for precision
            seq = np.array(seq, dtype=np.float32)
            all_deltas.append(seq)
            first_word = text.split()[0] if len(text.split()) > 0 else ""
            len_first_word = len(first_word)
            len_text = len(text)
            self.len_first_word_list.append(len_first_word)
            self.len_text_list.append(len_text)
            self.samples.append((text, seq, len_first_word, len_text))
        
        # Compute normalization statistics for starting coordinates
        self.start_coords = np.array(self.start_coords, dtype=np.float32)
        self.start_coord_mean = np.mean(self.start_coords, axis=0, dtype=np.float32)
        self.start_coord_std = np.std(self.start_coords, axis=0, dtype=np.float32) + 1e-6
        self.start_coord_std = np.maximum(self.start_coord_std, 1e-6)

        # Build normalization statistics with high precision
        # Only normalize dx, dy, dt - keep pressure and tilt in original range
        all_concat = np.concatenate(all_deltas, axis=0)
        self.mean = np.mean(all_concat[:, :3], axis=0, dtype=np.float32)
        self.std = np.std(all_concat[:, :3], axis=0, dtype=np.float32) + 1e-6
        
        # Ensure std is not zero to maintain precision during normalization
        self.std = np.maximum(self.std, 1e-6)

        # Build char vocabulary
        chars = sorted(list({c for entry in self.entries for c in entry["entry_text"]}))
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.char2idx)

        # Normalize length features
        self.len_first_word_arr = np.array(self.len_first_word_list, dtype=np.float32) if len(self.len_first_word_list) > 0 else np.array([0.0], dtype=np.float32)
        self.len_text_arr = np.array(self.len_text_list, dtype=np.float32) if len(self.len_text_list) > 0 else np.array([0.0], dtype=np.float32)
        self.len_first_word_mean = float(self.len_first_word_arr.mean())
        self.len_first_word_std = float(self.len_first_word_arr.std() + 1e-6)
        self.len_text_mean = float(self.len_text_arr.mean())
        self.len_text_std = float(self.len_text_arr.std() + 1e-6)

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return np.array([self.char2idx.get(c, 0) for c in text], dtype=np.int64)

    def __getitem__(self, idx):
        text, seq, len_first_word, len_text = self.samples[idx]
        
        # Normalize only dx, dy, dt - preserve pressure and tilt precision
        seq_norm = seq.copy().astype(np.float32)
        seq_norm[:, :3] = (seq[:, :3] - self.mean) / self.std
        
        # Create input (all but last) and target (all but first)
        # This matches the generation approach: given all previous steps [p0, p1, ..., pN-1],
        # predict the next steps [p1, p2, ..., pN]
        # The model will process the FULL input sequence, allowing it to use all previous context
        x = seq_norm[:-1].astype(np.float32)  # Input: all but last point
        y = seq_norm[1:].astype(np.float32)   # Target: all but first point (shifted by 1)
        
        # Add finished label: 1 for last point, 0 for all others
        # y has shape (seq_len-1, 5), we need to add finished dimension
        finished = np.zeros((len(y), 1), dtype=np.float32)
        finished[-1, 0] = 1.0  # Last point is finished
        y = np.concatenate([y, finished], axis=1)  # (seq_len-1, 6)

        # Progress fraction target per timestep in (0,1]
        total_steps = len(seq)
        if total_steps > 1:
            indices = np.arange(1, total_steps, dtype=np.float32)
            progress = (indices / float(total_steps)).reshape(-1, 1).astype(np.float32)
        else:
            progress = np.zeros((len(y), 1), dtype=np.float32)
        
        text_encoded = self.encode_text(text)

        # Inputs for restricted start coordinate model
        first_char = text[0] if len(text) > 0 else ""
        first_char_idx = self.char2idx.get(first_char, 0)
        len_first_word_norm = (float(len_first_word) - self.len_first_word_mean) / self.len_first_word_std
        len_text_norm = (float(len_text) - self.len_text_mean) / self.len_text_std
        
        # Get starting coordinates (normalized)
        start_coord = self.start_coords[idx]
        start_coord_norm = (start_coord - self.start_coord_mean) / self.start_coord_std
        
        return {
            'text': text_encoded,
            'stroke_input': x,
            'stroke_target': y,
            'start_coord': start_coord_norm.astype(np.float32),
            'progress_target': progress.astype(np.float32),
            'first_char_idx': np.int64(first_char_idx),
            'len_features': np.array([len_first_word_norm, len_text_norm], dtype=np.float32)
        }


# ---------- COLLATE FUNCTION FOR PADDING ----------
def collate_fn(batch):
    """Collate function to pad sequences to same length within a batch"""
    texts = [item['text'] for item in batch]
    stroke_inputs = [item['stroke_input'] for item in batch]
    stroke_targets = [item['stroke_target'] for item in batch]
    progress_targets = [item['progress_target'] for item in batch]
    
    # Get max lengths
    max_text_len = max(len(t) for t in texts)
    max_stroke_len = max(len(s) for s in stroke_inputs)
    
    # Pad texts
    padded_texts = []
    for text in texts:
        padded = np.pad(text, (0, max_text_len - len(text)), mode='constant', constant_values=0)
        padded_texts.append(padded)
    
    # Pad stroke sequences
    padded_stroke_inputs = []
    padded_stroke_targets = []
    for stroke_in, stroke_tgt in zip(stroke_inputs, stroke_targets):
        # Pad with zeros
        pad_len_in = max_stroke_len - len(stroke_in)
        pad_len_tgt = max_stroke_len - len(stroke_tgt)
        
        padded_in = np.pad(stroke_in, ((0, pad_len_in), (0, 0)), mode='constant', constant_values=0.0)
        # For targets, pad with zeros (finished will be 0 for padding)
        padded_tgt = np.pad(stroke_tgt, ((0, pad_len_tgt), (0, 0)), mode='constant', constant_values=0.0)
        
        padded_stroke_inputs.append(padded_in)
        padded_stroke_targets.append(padded_tgt)

    # Pad progress targets to match sequence length
    padded_progress_targets = []
    for prog in progress_targets:
        pad_len_prog = max_stroke_len - len(prog)
        padded_prog = np.pad(prog, ((0, pad_len_prog), (0, 0)), mode='constant', constant_values=0.0)
        padded_progress_targets.append(padded_prog)
    
    # Convert to tensors
    text_tensor = torch.from_numpy(np.array(padded_texts)).long()
    stroke_input_tensor = torch.from_numpy(np.array(padded_stroke_inputs)).float()
    stroke_target_tensor = torch.from_numpy(np.array(padded_stroke_targets)).float()
    progress_target_tensor = torch.from_numpy(np.array(padded_progress_targets)).float()
    
    # Create mask for valid positions (not padding)
    # Mask is True where stroke_input is not all zeros
    mask = torch.any(stroke_input_tensor != 0.0, dim=-1)  # (batch, seq_len)
    
    # Get starting coordinates
    start_coords = torch.from_numpy(np.array([item['start_coord'] for item in batch])).float()
    first_char_idx_tensor = torch.from_numpy(np.array([item['first_char_idx'] for item in batch])).long()
    len_features_tensor = torch.from_numpy(np.array([item['len_features'] for item in batch])).float()
    
    return {
        'text': text_tensor,
        'stroke_input': stroke_input_tensor,
        'stroke_target': stroke_target_tensor,
        'mask': mask,
        'start_coord': start_coords,
        'progress_target': progress_target_tensor,
        'first_char_idx': first_char_idx_tensor,
        'len_features': len_features_tensor
    }


# ---------- MODEL ----------
class TextConditionedDecoder(nn.Module):
    def __init__(self, vocab_size, input_size=5, embed_dim=256, hidden_size=1024, num_layers=4, output_size=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size  # 6: [dx, dy, dt, pressure, tilt, finished]
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        # Use multi-layer LSTM for better capacity
        self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        # Add intermediate layer for better feature extraction
        self.fc_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, stroke_seq):
        """
        Args:
            text: (B, T_text) - text token indices
            stroke_seq: (B, T_stroke, 5) - stroke sequence
        Returns:
            output: (B, T_stroke, 6) - predicted stroke sequence [dx, dy, dt, pressure, tilt, finished]
                 finished is sigmoid activated (0-1 range)
        """
        # Encode text and average embeddings to form context vector
        text_emb = self.embedding(text)  # (B, T_text, E)
        
        # Mask out padding tokens (index 0) when averaging
        # Create mask: True where token is not padding (not 0)
        text_mask = (text != 0).float()  # (B, T_text)
        text_mask = text_mask.unsqueeze(-1)  # (B, T_text, 1)
        
        # Apply mask and average
        masked_emb = text_emb * text_mask  # (B, T_text, E)
        mask_sum = text_mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
        context = masked_emb.sum(dim=1, keepdim=True) / (mask_sum + 1e-8)  # (B, 1, E)
        
        # Repeat context for each timestep in stroke sequence
        seq_len = stroke_seq.size(1)
        context = context.repeat(1, seq_len, 1)  # (B, seq_len, E)
        
        # Concatenate stroke sequence with context
        combined = torch.cat([stroke_seq, context], dim=-1)  # (B, seq_len, input_size + embed_dim)
        
        # Pass through LSTM
        out, _ = self.lstm(combined)  # (B, seq_len, hidden_size)
        
        # Pass through intermediate layer
        out = self.fc_hidden(out)  # (B, seq_len, hidden_size // 2)
        out = self.relu(out)
        
        # Final output layer
        out = self.fc(out)  # (B, seq_len, output_size)
        
        # Apply sigmoid to finished (last dimension)
        # Split output: first 5 dims are stroke features, last dim is finished
        stroke_features = out[:, :, :5]  # (B, seq_len, 5)
        finished_logit = out[:, :, 5:6]  # (B, seq_len, 1)
        finished = self.sigmoid(finished_logit)  # (B, seq_len, 1) - sigmoid activated
        
        # Concatenate back
        out = torch.cat([stroke_features, finished], dim=-1)  # (B, seq_len, 6)
        
        return out


# ---------- DNN MODEL FOR STARTING COORDINATES ----------
class StartCoordPredictor(nn.Module):
    """DNN model to predict starting coordinates (x, y) from restricted inputs"""
    def __init__(self, vocab_size, embed_dim=32, hidden_dims=[128, 64], output_size=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_size = output_size  # 2 for (x, y)
        
        # Embed first character only
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # Build DNN layers
        layers = []
        input_dim = embed_dim + 2
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, first_char_idx, len_features):
        """
        Args:
            first_char_idx: (B,) - token index of first character
            len_features: (B, 2) - [len_first_word_norm, len_text_norm]
        Returns:
            coords: (B, 2) - predicted starting coordinates (x, y) - normalized
        """
        first_char_emb = self.embedding(first_char_idx)  # (B, E)
        features = torch.cat([first_char_emb, len_features], dim=-1)  # (B, E+2)
        coords = self.network(features)  # (B, 2)
        return coords


# ---------- FINISHED PROGRESS MODEL ----------
class FinishedProgressPredictor(nn.Module):
    """Predicts fraction of strokes completed at each timestep given text and stroke input."""
    def __init__(self, vocab_size, input_size=5, embed_dim=64, hidden_size=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, stroke_seq):
        text_emb = self.embedding(text)
        text_mask = (text != 0).float().unsqueeze(-1)
        masked_emb = text_emb * text_mask
        mask_sum = text_mask.sum(dim=1, keepdim=True)
        context = masked_emb.sum(dim=1, keepdim=True) / (mask_sum + 1e-8)
        seq_len = stroke_seq.size(1)
        context_rep = context.repeat(1, seq_len, 1)
        combined = torch.cat([stroke_seq, context_rep], dim=-1)
        out, _ = self.lstm(combined)
        out = self.fc(out)
        progress = self.sigmoid(out)
        return progress  # (B, T, 1)


# ---------- LOSS FUNCTION ----------
def masked_mse_loss(y_pred, y_true, mask):
    """
    Compute combined loss with masking for variable-length sequences.
    Uses MSE for stroke features and BCE for finished parameter.
    
    Args:
        y_pred: (B, T, 6) - predicted strokes [dx, dy, dt, pressure, tilt, finished]
        y_true: (B, T, 6) - target strokes
        mask: (B, T) - boolean mask, True for valid positions
    Returns:
        loss: scalar
    """
    # Expand mask to match feature dimensions
    mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
    
    # Split predictions and targets
    stroke_pred = y_pred[:, :, :5]  # (B, T, 5) - stroke features
    finished_pred = y_pred[:, :, 5:6]  # (B, T, 1) - finished (sigmoid activated)
    
    stroke_true = y_true[:, :, :5]  # (B, T, 5)
    finished_true = y_true[:, :, 5:6]  # (B, T, 1)
    
    # MSE loss for stroke features
    stroke_squared_diff = (stroke_pred - stroke_true) ** 2  # (B, T, 5)
    stroke_masked_diff = stroke_squared_diff * mask_expanded  # (B, T, 5)
    stroke_loss = stroke_masked_diff.sum() / (mask_expanded.sum() + 1e-8)
    
    # BCE loss for finished parameter (binary classification)
    # Use mask to only compute loss on valid positions
    finished_mask = mask.unsqueeze(-1).float()  # (B, T, 1)
    finished_bce = nn.functional.binary_cross_entropy(
        finished_pred, 
        finished_true, 
        reduction='none'
    )  # (B, T, 1)
    finished_masked = finished_bce * finished_mask
    finished_loss = finished_masked.sum() / (finished_mask.sum() + 1e-8)
    
    # Combine losses (weight finished loss more heavily to ensure it learns)
    loss = stroke_loss + 2.0 * finished_loss
    
    return loss


def masked_regression_loss(y_pred, y_true, mask):
    """MSE regression loss for sequences with mask."""
    mask_expanded = mask.unsqueeze(-1).float()
    sq = (y_pred - y_true) ** 2
    sq_masked = sq * mask_expanded
    return sq_masked.sum() / (mask_expanded.sum() + 1e-8)


# ---------- TRAIN ----------
def train():
    # Load dataset
    dataset_obj = HandwritingDataset(DATA_FILE)
    vocab_size = dataset_obj.vocab_size
    
    print(f"Dataset loaded: {len(dataset_obj)} samples")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset_obj,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Build models
    model = TextConditionedDecoder(vocab_size)
    model = model.to(device)
    
    start_coord_model = StartCoordPredictor(vocab_size)
    start_coord_model = start_coord_model.to(device)
    finished_progress_model = FinishedProgressPredictor(vocab_size)
    finished_progress_model = finished_progress_model.to(device)
    
    # Count parameters for both models
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    start_coord_params = sum(p.numel() for p in start_coord_model.parameters())
    finished_progress_params = sum(p.numel() for p in finished_progress_model.parameters())
    
    print(f"RNN Model - Total parameters: {total_params:,}")
    print(f"RNN Model - Trainable parameters: {trainable_params:,}")
    print(f"Start Coord DNN - Total parameters: {start_coord_params:,}")
    print(f"Finished Progress Model - Total parameters: {finished_progress_params:,}")
    
    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=LR)
    start_coord_optimizer = optim.Adam(start_coord_model.parameters(), lr=LR)
    finished_progress_optimizer = optim.Adam(finished_progress_model.parameters(), lr=LR)
    
    print(f"Training | samples={len(dataset_obj)} | epochs={EPOCHS}")
    
    # Training loop
    model.train()
    start_coord_model.train()
    finished_progress_model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_start_coord_loss = 0.0
        total_progress_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            text_batch = batch['text'].to(device)
            stroke_input_batch = batch['stroke_input'].to(device)
            stroke_target_batch = batch['stroke_target'].to(device)
            mask = batch['mask'].to(device)
            start_coord_batch = batch['start_coord'].to(device)
            progress_target_batch = batch['progress_target'].to(device)
            first_char_idx_batch = batch['first_char_idx'].to(device)
            len_features_batch = batch['len_features'].to(device)
            
            # ========== Train RNN model ==========
            optimizer.zero_grad()
            
            # Forward pass
            # stroke_input_batch contains the FULL sequence for each sample (all previous steps)
            # The LSTM processes the entire sequence and maintains context across all timesteps
            # This matches generation where we feed all previous steps to predict the next one
            preds = model(text_batch, stroke_input_batch)
            
            # Compute masked loss
            loss = masked_mse_loss(preds, stroke_target_batch, mask)
            
            # Monitor finished parameter (for debugging)
            finished_pred = preds[:, :, 5:6]
            finished_true = stroke_target_batch[:, :, 5:6]
            valid_mask = mask.unsqueeze(-1).float()
            finished_mean = (finished_pred * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            finished_target_mean = (finished_true * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            
            # Backward pass
            loss.backward()
            # Clip gradients to prevent exploding gradients and mode collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # ========== Train Start Coord DNN model ==========
            start_coord_optimizer.zero_grad()
            
            # Forward pass
            pred_start_coords = start_coord_model(first_char_idx_batch, len_features_batch)
            
            # Compute MSE loss for starting coordinates
            start_coord_loss = nn.functional.mse_loss(pred_start_coords, start_coord_batch)
            
            # Backward pass
            start_coord_loss.backward()
            torch.nn.utils.clip_grad_norm_(start_coord_model.parameters(), max_norm=1.0)
            start_coord_optimizer.step()

            # ========== Train Finished Progress model ==========
            finished_progress_optimizer.zero_grad()
            progress_pred = finished_progress_model(text_batch, stroke_input_batch)
            progress_loss = masked_regression_loss(progress_pred, progress_target_batch, mask)
            progress_loss.backward()
            torch.nn.utils.clip_grad_norm_(finished_progress_model.parameters(), max_norm=1.0)
            finished_progress_optimizer.step()
            
            # Update metrics
            loss_val = loss.item()
            start_coord_loss_val = start_coord_loss.item()
            progress_loss_val = progress_loss.item()
            finished_mean_val = finished_mean.item()
            finished_target_mean_val = finished_target_mean.item()
            total_loss += loss_val
            total_start_coord_loss += start_coord_loss_val
            total_progress_loss += progress_loss_val
            num_batches += 1
            pbar.set_postfix({
                'rnn_loss': f'{loss_val:.6f}',
                'start_loss': f'{start_coord_loss_val:.6f}',
                'progress_loss': f'{progress_loss_val:.6f}',
                'finished_pred': f'{finished_mean_val:.4f}',
                'finished_true': f'{finished_target_mean_val:.4f}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_start_coord_loss = total_start_coord_loss / num_batches if num_batches > 0 else 0.0
        avg_progress_loss = total_progress_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}: RNN Loss={avg_loss:.6f}, Start Coord Loss={avg_start_coord_loss:.6f}, Progress Loss={avg_progress_loss:.6f}")
    
    # Save model and metadata
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': vocab_size,
            'input_size': 5,
            'embed_dim': 256,
            'hidden_size': 1024,
            'num_layers': 4,
            'output_size': 6  # [dx, dy, dt, pressure, tilt, finished]
        }
    }, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    
    # Save metadata separately
    metadata_path = MODEL_PATH.replace(".pt", "_metadata.json")
    metadata = {
        "char2idx": dataset_obj.char2idx,
        "idx2char": {str(k): v for k, v in dataset_obj.idx2char.items()},
        "mean": dataset_obj.mean.tolist(),
        "std": dataset_obj.std.tolist(),
        "vocab_size": vocab_size
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {metadata_path}")
    
    # Save DNN model for starting coordinates
    torch.save({
        'model_state_dict': start_coord_model.state_dict(),
        'model_config': {
            'vocab_size': vocab_size,
            'embed_dim': 32,
            'hidden_dims': [128, 64],
            'output_size': 2
        },
        'start_coord_mean': dataset_obj.start_coord_mean.tolist(),
        'start_coord_std': dataset_obj.start_coord_std.tolist(),
        'len_first_word_mean': dataset_obj.len_first_word_mean,
        'len_first_word_std': dataset_obj.len_first_word_std,
        'len_text_mean': dataset_obj.len_text_mean,
        'len_text_std': dataset_obj.len_text_std
    }, START_COORD_MODEL_PATH)
    print(f"✅ Start Coord DNN model saved to {START_COORD_MODEL_PATH}")

    # Save Finished Progress model
    torch.save({
        'model_state_dict': finished_progress_model.state_dict(),
        'model_config': {
            'vocab_size': vocab_size,
            'input_size': 5,
            'embed_dim': 64,
            'hidden_size': 256,
            'num_layers': 1,
            'output_size': 1
        }
    }, FINISHED_PROGRESS_MODEL_PATH)
    print(f"✅ Finished Progress model saved to {FINISHED_PROGRESS_MODEL_PATH}")
    
    # Update metadata to include start coord normalization stats
    metadata['start_coord_mean'] = dataset_obj.start_coord_mean.tolist()
    metadata['start_coord_std'] = dataset_obj.start_coord_std.tolist()
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Updated metadata with start coord normalization stats")


if __name__ == "__main__":
    train()