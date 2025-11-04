# =============================================
# train_handwriting_rnn_v2.py - TensorFlow Version
# =============================================
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import os

# Set precision mode for high precision computation
tf.keras.backend.set_floatx('float32')

# ---------- PARAMETERS ----------
EPOCHS = 20
BATCH_SIZE = 2
LR = 1e-3
MODEL_PATH = "my_model.keras"
DATA_FILE = "input_data.json"

# ---------- DATASET ----------
class HandwritingDataset:
    def __init__(self, filename):
        with open(filename) as f:
            self.entries = json.load(f)

        self.samples = []
        all_deltas = []

        # Parse each entry - keep ALL raw points without merging
        for entry in self.entries:
            text = entry["entry_text"]
            stroke_data = entry["stroke_data"]
            # Sort points by timestamp to maintain temporal order
            points = sorted(stroke_data.values(), key=lambda p: p["timestamp"])
            if len(points) < 2:
                continue
            
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
            
            # Use float32 for precision (can be changed to float64 if needed)
            seq = np.array(seq, dtype=np.float32)
            all_deltas.append(seq)
            self.samples.append((text, seq))

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

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return np.array([self.char2idx.get(c, 0) for c in text], dtype=np.int32)

    def get_samples(self):
        """Return all samples as lists for tf.data.Dataset"""
        texts = []
        xs = []
        ys = []
        
        for text, seq in self.samples:
            # Normalize only dx, dy, dt - preserve pressure and tilt precision
            seq_norm = seq.copy().astype(np.float32)
            seq_norm[:, :3] = (seq[:, :3] - self.mean) / self.std
            
            # Create input (all but last) and target (all but first)
            x = seq_norm[:-1].astype(np.float32)
            y = seq_norm[1:].astype(np.float32)
            
            texts.append(self.encode_text(text))
            xs.append(x)
            ys.append(y)
        
        return texts, xs, ys


# ---------- MODEL ----------
class TextConditionedDecoder(keras.Model):
    def __init__(self, vocab_size, input_size=5, embed_dim=64, hidden_size=256, output_size=5, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = keras.layers.Embedding(vocab_size + 1, embed_dim, mask_zero=True)
        self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=False)
        self.fc = keras.layers.Dense(output_size)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "input_size": self.input_size,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, inputs, training=None):
        text, stroke_seq = inputs
        
        # Encode text and average embeddings to form context vector
        text_emb = self.embedding(text)  # (B, T, E)
        
        # Mask out padding tokens (index 0) when averaging
        # Get mask from embedding layer
        mask = self.embedding.compute_mask(text)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)  # (B, T, 1)
            masked_emb = text_emb * mask
            # Average over sequence length, excluding padding
            context = tf.reduce_sum(masked_emb, axis=1, keepdims=True) / (
                tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-8
            )  # (B, 1, E)
        else:
            # If no mask, just average
            context = tf.reduce_mean(text_emb, axis=1, keepdims=True)  # (B, 1, E)
        
        # Repeat context for each timestep in stroke sequence
        seq_len = tf.shape(stroke_seq)[1]
        context = tf.repeat(context, seq_len, axis=1)  # (B, seq_len, E)
        
        # Concatenate stroke sequence with context
        combined = tf.concat([stroke_seq, context], axis=-1)  # (B, seq_len, input_size + embed_dim)
        
        # Pass through LSTM
        out = self.lstm(combined, training=training)
        
        # Final output layer
        out = self.fc(out)
        return out


# ---------- DATA PIPELINE ----------
def create_dataset(texts, xs, ys, batch_size=2, shuffle=True):
    """Create a tf.data.Dataset from lists of samples"""
    
    def generator():
        for text, x, y in zip(texts, xs, ys):
            yield (text, x), y
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec(shape=(None,), dtype=tf.int32),  # text
             tf.TensorSpec(shape=(None, 5), dtype=tf.float32)),  # x
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32)  # y
        )
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Pad sequences to same length within batches
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (tf.TensorShape([None]),  # text
             tf.TensorShape([None, 5])),  # x
            tf.TensorShape([None, 5])  # y
        ),
        padding_values=(
            (0, 0.0),  # pad text with 0, pad x with 0.0
            0.0  # pad y with 0.0
        ),
        drop_remainder=False
    )
    
    return dataset


# ---------- LOSS FUNCTION ----------
def masked_mse_loss(y_true, y_pred, sample_weight=None):
    """Compute MSE loss with masking for variable-length sequences"""
    # Compute squared difference
    squared_diff = tf.square(y_pred - y_true)
    
    # If sample_weight is provided (mask), use it
    if sample_weight is not None:
        squared_diff = squared_diff * sample_weight
        loss = tf.reduce_sum(squared_diff) / (tf.reduce_sum(sample_weight) + 1e-8)
    else:
        loss = tf.reduce_mean(squared_diff)
    
    return loss


# ---------- TRAIN ----------
def train():
    # Load dataset
    dataset_obj = HandwritingDataset(DATA_FILE)
    vocab_size = dataset_obj.vocab_size
    
    print(f"Dataset loaded: {len(dataset_obj)} samples")
    print(f"Vocabulary size: {vocab_size}")
    
    # Get samples
    texts, xs, ys = dataset_obj.get_samples()
    
    # Create tf.data.Dataset
    train_dataset = create_dataset(texts, xs, ys, batch_size=BATCH_SIZE, shuffle=True)
    
    # Build model
    model = TextConditionedDecoder(vocab_size)
    
    # Build model with dummy inputs to initialize weights
    # This allows us to count parameters
    dummy_text = tf.constant([[1] * 10], dtype=tf.int32)  # Dummy text sequence
    dummy_stroke = tf.zeros((1, 1, 5), dtype=tf.float32)  # Dummy stroke sequence
    _ = model([dummy_text, dummy_stroke], training=False)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    
    # Create custom loss that handles masking
    def compute_loss(y_true, y_pred, x_batch):
        """Compute loss with proper masking for padded sequences"""
        # Create mask: True where x_batch is not all zeros (i.e., not padding)
        # Check if any feature is non-zero
        mask = tf.reduce_any(tf.not_equal(x_batch, 0.0), axis=-1)  # (batch, seq_len)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)  # (batch, seq_len, 1)
        
        # Compute MSE with masking
        squared_diff = tf.square(y_pred - y_true)
        masked_squared_diff = squared_diff * mask
        loss = tf.reduce_sum(masked_squared_diff) / (tf.reduce_sum(mask) + 1e-8)
        
        return loss
    
    print(f"Training | samples={len(dataset_obj)} | epochs={EPOCHS}")
    print(f"Model parameters: {model.count_params():,}")
    
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        pbar = tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            (text_batch, x_batch), y_batch = batch
            
            with tf.GradientTape() as tape:
                # Forward pass
                preds = model([text_batch, x_batch], training=True)
                
                # Compute masked loss
                loss = compute_loss(y_batch, preds, x_batch)
            
            # Compute gradients and update weights
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            loss_val = loss.numpy()
            total_loss += loss_val
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss_val:.6f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")
    
    # Save model and metadata
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save Keras model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    
    # Save metadata separately
    metadata_path = MODEL_PATH.replace(".keras", "_metadata.json")
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


if __name__ == "__main__":
    # Set TensorFlow to use GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"✅ Using GPU: {physical_devices[0]}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Using CPU")
    
    train()
