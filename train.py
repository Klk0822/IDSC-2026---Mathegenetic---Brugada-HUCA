import os
import hashlib
import warnings
import glob
from typing import Tuple, Dict, List

# AI models use a lot of random numbers to learn. By setting the "seed" to a specific number (like 0),
# we guarantee that the code will do the exact same thing every time we run it.
os.environ['PYTHONHASHSEED'] = "0"

import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics, regularizers

# =====================================================================
# GLOBAL SETTINGS & FILE PATHS
# =====================================================================
# Think of this section as the control panel for the whole script. 
# Changing a value here updates it everywhere in the code.
RANDOM_SEED = 2026
BASE_DIR = r"C:\Users\User\Downloads\IDSC-2026---Mathegenetic---Brugada-HUCA-main"
FILES_DIR = os.path.join(BASE_DIR, "files")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")

# A standard ECG test uses 12 "leads" (sensors placed on the body).
STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 100 # We record 100 data points every second.
TIMESTEPS = 1200 # We will look at exactly 1200 data points (12 seconds) for every patient.
NUM_RAW_LEADS = len(STANDARD_LEADS)

# For every lead, we create 3 versions of the data to help the AI learn better.
# 12 leads * 3 versions = 36 total channels.
NUM_MODEL_CHANNELS = NUM_RAW_LEADS * 3  
CUTOFF_FREQ = 0.5
MODEL_SAVE_NAME = "brugada_champion_model.keras"
SCALERS_SAVE_NAME = "brugada_scalers.npz"


# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================
def setup_environment(seed: int = RANDOM_SEED):
    """Sets up the computer so it's ready to train the AI model smoothly."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Sometimes Python prints out annoying warnings that don't actually affect our code. 
    # This tells Python to stay quiet and ignore them.
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)

    # If your computer has a dedicated Graphics Card (GPU), this tells the AI to use it 
    # efficiently so it doesn't run out of memory and crash.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")


# =====================================================================
# DATA LOADING & PREPARATION
# =====================================================================
def load_data(files_dir: str, metadata_file: str) -> Tuple[pd.DataFrame, dict]:
    """Loads the patient spreadsheet (metadata) and their actual ECG heartbeat readings."""
    md = pd.read_csv(metadata_file)
    md['patient_id'] = md['patient_id'].astype(str).str.strip()
    
    # We want a simple Yes/No answer. 1 = Has Brugada, 0 = Does not have Brugada.
    md.loc[md['brugada'] == 2, 'brugada'] = 1
    
    # Search through all the folders to find the ECG heartbeat files
    hea_files = glob.glob(os.path.join(files_dir, "**", "*.hea"), recursive=True)
    ecg_dict = {}
    failed_files = []
    
    # Open each file and store the heartbeat data in a dictionary
    for path in [os.path.splitext(f)[0] for f in hea_files]:
        pid = str(os.path.basename(path)).strip()
        try:
            record = wfdb.rdrecord(path)
            ecg_dict[pid] = pd.DataFrame(record.p_signal, columns=record.sig_name)
        except Exception:
            failed_files.append(pid)
            
    if failed_files:
        print(f"Warning: Failed to load {len(failed_files)} files.")
        
    return md, ecg_dict


def _filter_and_crop_signal(raw_signal: np.ndarray, b: np.ndarray, a: np.ndarray, min_len: int) -> np.ndarray:
    """Helper function: Cleans up a single heartbeat recording and trims it to the right length."""
    clean_signal = raw_signal[~np.isnan(raw_signal)]
    
    # 1. Clean the signal: We use a digital filter to remove unwanted noise 
    # (like the patient breathing or moving around) from the heartbeat line.
    if len(clean_signal) > max(min_len, TIMESTEPS) and np.std(clean_signal) > 1e-8:
        filled_signal = pd.Series(raw_signal).interpolate(limit_direction='both').to_numpy()
        filled_signal = np.nan_to_num(filled_signal, nan=0.0) 
        
        if np.std(filled_signal) < 1e-8:
            filtered = filled_signal
        else:
            filtered = filtfilt(b, a, filled_signal)
            
        filtered[np.isnan(raw_signal)] = np.nan 
    else:
        filtered = raw_signal
        
    # 2. Crop to the center: Heartbeat recordings are all different lengths. 
    # AI models prefer data that is all the exact same size. So, we grab exactly 
    # 1200 data points from the dead-center of the recording.
    sig_len = len(filtered)
    if sig_len > TIMESTEPS:
        start = max(0, (sig_len - TIMESTEPS) // 2)
        return filtered[start:start+TIMESTEPS]
    
    # If the recording is too short, we pad it with empty values to make it 1200 long.
    padded = np.full(TIMESTEPS, np.nan)
    padded[:sig_len] = filtered
    return padded


def preprocess_and_align(ecg_dict: dict, md: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Applies the cleaning process to every patient and matches them with their Yes/No Brugada label."""
    nyquist = FS / 2
    b, a = butter(N=3, Wn=CUTOFF_FREQ / nyquist, btype="high")
    min_len = 3 * max(len(a), len(b))
    
    valid_ids = [pid for pid in ecg_dict.keys() if pid in md['patient_id'].values]
    
    # Create a giant empty grid to hold all the cleaned heartbeats
    X_array = np.full((len(valid_ids), TIMESTEPS, NUM_RAW_LEADS), np.nan)
    
    # Fill the grid with the cleaned data
    for i, pid in enumerate(valid_ids):
        df = ecg_dict[pid].copy()
        
        for lead_idx, lead_name in enumerate(STANDARD_LEADS):
            if lead_name not in df.columns:
                X_array[i, :, lead_idx] = np.nan
                continue
                
            raw_signal = df[lead_name].values
            X_array[i, :, lead_idx] = _filter_and_crop_signal(raw_signal, b, a, min_len)
                
    # Create a list of the answers (1 for Brugada, 0 for healthy) so the AI can learn from it
    aligned_md = md.set_index('patient_id').reindex(valid_ids).dropna(subset=['brugada']).reset_index()
    Y_target = aligned_md['brugada'].values.astype(int)
    
    return X_array, Y_target


# =====================================================================
# DATA SCALING (Making it easy for the AI to read)
# =====================================================================
def normalize_per_sample_safe(X: np.ndarray) -> np.ndarray:
    """Adjusts each heartbeat so it isn't too loud or too quiet. It levels the playing field."""
    X_norm = np.copy(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            lead = X[i, :, j]
            valid = ~np.isnan(lead)
            if np.sum(valid) > 10:
                mean = np.mean(lead[valid])
                std = np.std(lead[valid])
                std = std if std > 1e-8 else 1.0
                # Subtract the average and divide by the spread
                X_norm[i, valid, j] = (lead[valid] - mean) / std
                
    return np.nan_to_num(X_norm, nan=0.0)

def fit_global_amplitude_scalers(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Looks at all the training data to figure out the average size of a heartbeat."""
    X_flat = X_train.reshape(-1, NUM_RAW_LEADS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(X_flat, axis=0)
        stds = np.nanstd(X_flat, axis=0)
        
    means = np.nan_to_num(means, nan=0.0)
    stds = np.where(np.nan_to_num(stds, nan=1.0) > 1e-8, stds, 1.0)
    return means, stds

def apply_global_amplitude_scalers(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Applies that global average to new data."""
    means = means.reshape(1, 1, -1)
    stds = stds.reshape(1, 1, -1)
    X_amp = (X - means) / stds
    return np.nan_to_num(X_amp, nan=0.0)

def build_composite_tensor(X_raw: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Packages the data nicely. It creates 3 layers of info for the AI:
       1. The shape of the heartbeat.
       2. The size of the heartbeat.
       3. A map of where data is missing."""
    mask = (~np.isnan(X_raw)).astype(float)
    X_norm = normalize_per_sample_safe(X_raw)
    X_amp = apply_global_amplitude_scalers(X_raw, means, stds)
    
    # Replace missing data with zeros so the AI doesn't break
    X_norm[np.isnan(X_raw)] = 0.0
    X_amp[np.isnan(X_raw)] = 0.0
    
    # Glue them all together
    return np.concatenate([X_norm, X_amp, mask], axis=2)


# =====================================================================
# THE AI MODEL ARCHITECTURE (The "Brain")
# =====================================================================
def binary_focal_loss(gamma=2.0, alpha=0.5):
    """
    This is how the model is graded. Instead of treating all mistakes equally, 
    this special 'focal loss' forces the model to pay extra attention to the 
    really difficult cases that it keeps getting wrong.
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return -tf.reduce_mean(alpha_factor * modulating_factor * tf.math.log(p_t))
    return focal_loss_fn

def se_block(tensor, ratio=8):
    """Helps the AI figure out which of the 12 ECG sensors are the most important right now."""
    channels = int(tensor.shape[-1])
    se = layers.GlobalAveragePooling1D()(tensor)
    se = layers.Dense(channels // ratio, activation='swish', kernel_initializer='he_normal')(se)
    se = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = layers.Reshape((1, channels))(se)
    return layers.Multiply()([tensor, se])

def res_block_1d(x, filters, kernel_size):
    """
    A building block for the neural network. It includes a 'shortcut' connection 
    so that as the network gets deeper, it doesn't forget the information it learned earlier.
    """
    shortcut = layers.Conv1D(filters, 1, padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.SpatialDropout1D(0.1)(x) # Randomly drop data to prevent the AI from memorizing the answers

    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.Add()([x, shortcut]) # Here is where the shortcut connects!
    x = layers.Activation('swish')(x)
    x = layers.MaxPooling1D(pool_size=2)(x) # Shrink the data to make processing faster
    return x

def build_clinical_model() -> models.Model:
    """Assembles all the building blocks into the final AI 'Brain'."""
    inputs = layers.Input(shape=(TIMESTEPS, NUM_MODEL_CHANNELS))
    
    # Add a tiny bit of static (noise) so the AI learns to be tough and not rely on perfect data
    x = layers.GaussianNoise(0.05)(inputs) 
    
    # The layers that scan the heartbeat shapes
    x = res_block_1d(x, filters=64, kernel_size=15)
    x = res_block_1d(x, filters=128, kernel_size=7)
    x = res_block_1d(x, filters=256, kernel_size=3)
    
    x = layers.Activation('linear', name='final_conv')(x)
    
    # The layers that understand the sequence and timing of the heartbeat
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, reset_after=False))(x)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    # Compress all that learned info down
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(rate=0.4)(x)
    x = layers.Dense(64, activation='swish')(x)
    x = layers.Dropout(rate=0.4)(x)
    
    # The final decision! It outputs a percentage from 0% to 100% chance of having Brugada.
    outputs = layers.Dense(units=1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Give the model its rules for how to update and learn from its mistakes (Adam optimizer)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss=binary_focal_loss(gamma=2.0, alpha=0.5),
                  metrics=[metrics.AUC(name='auc')])
    return model


# =====================================================================
# EVALUATION & GRADING
# =====================================================================
def split_dataset(X_raw: np.ndarray, y_target: np.ndarray):
    """
    We can't test the AI on data it has already seen (that's cheating!). 
    So we split the patients into three groups:
    1. Training (80%): The AI studies these patients to learn.
    2. Validation (10%): A practice test to see how the AI is doing.
    3. Testing (10%): The final exam using data the AI has NEVER seen.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_target, test_size=0.10, stratify=y_target, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def optimize_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    The AI gives a percentage (like 45% chance of Brugada). 
    This function figures out the best cutoff point. Maybe anything over 40% 
    should be considered a "Yes" to be on the safe side.
    """
    thresholds = np.unique(y_pred_prob)
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh


def evaluate_and_print_metrics(model, X_test: np.ndarray, y_test: np.ndarray, optimal_thresh: float):
    """Scores the final exam and prints out a report card of how well the AI did."""
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= optimal_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2.0, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Test Accuracy:  {acc:.4f}\nTest Precision: {prec:.4f}\nTest Recall:    {rec:.4f}")
    print(f"Test F1 Score:  {f1:.4f}\nTest F2 Score:  {f2:.4f}\nTest AUC:       {roc_auc:.4f}")


# =====================================================================
# VISUALIZING HOW THE AI THINKS (Grad-CAM)
# =====================================================================
def compute_gradcam_1d(model, x_input_tensor, layer_name='final_conv') -> np.ndarray:
    """
    Grad-CAM acts like an X-ray for our AI model. 
    It looks backwards through the network to figure out exactly *which part* of the heartbeat made the model decide a patient has Brugada syndrome.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        tape.watch(x_input_tensor)
        conv_outputs, predictions = grad_model(x_input_tensor)
        loss = predictions[0, 0] 
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=1) 
    
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) # Only keep the positive signals
    
    if len(heatmap) == 0:
        return np.zeros(TIMESTEPS)
        
    max_heat = np.max(heatmap)
    if max_heat > 0:
        heatmap /= max_heat
        
    # Stretch the heatmap so it matches the length of the heartbeat recording
    f = interp1d(np.arange(len(heatmap)), heatmap, kind='linear', fill_value="extrapolate")
    return f(np.linspace(0, len(heatmap) - 1, TIMESTEPS))


def visualize_gradcam(model, X_test_raw, y_test, X_test_composite, optimal_thresh, means, stds):
    """Draws a pretty graph of the heartbeat and highlights the 'suspicious' areas in red for both V1 and V2 in separate pictures."""
    y_pred_prob = model.predict(X_test_composite, verbose=0).flatten()
    tp_idx = np.where((y_test == 1) & (y_pred_prob >= optimal_thresh))[0]
    
    if len(tp_idx) == 0:
        print("No True Positives identified in the Test set at the optimal threshold.")
        return
        
    print(f"Visualizing Saliency Map for True Positive Patient from Test Set...")
    sample_idx = tp_idx[0]
    tp_raw = X_test_raw[sample_idx:sample_idx+1]
    
    tp_tensor = tf.convert_to_tensor(build_composite_tensor(tp_raw, means, stds), dtype=tf.float32)
    heatmap = compute_gradcam_1d(model, tp_tensor)
    
    v1_lead_idx = STANDARD_LEADS.index('V1')
    v2_lead_idx = STANDARD_LEADS.index('V2')
    
    v1_signal = tp_raw[0, :, v1_lead_idx]
    v2_signal = tp_raw[0, :, v2_lead_idx]
    
    # ---------------------------------------------
    # PICTURE 1: Plotting Lead V1 in its own window
    # ---------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(v1_signal, color='black', label='Lead V1 Signal', linewidth=1.5)
    
    heat_scaled_v1 = (heatmap / np.max(heatmap)) * (np.nanmax(v1_signal) - np.nanmin(v1_signal)) + np.nanmin(v1_signal)
    plt.plot(heat_scaled_v1, color='red', label='Grad-CAM Attention', alpha=0.7, linewidth=2)
    plt.fill_between(range(TIMESTEPS), np.nanmin(v1_signal), heat_scaled_v1, color='red', alpha=0.2)
    
    plt.title('1D Grad-CAM Saliency Map: Brugada Detection Focus (Lead V1)', fontweight='bold')
    plt.xlabel('Time Steps (10ms / step)')
    plt.ylabel('Amplitude (mV)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show() # This triggers the first popup window

    # ---------------------------------------------
    # PICTURE 2: Plotting Lead V2 in its own window
    # ---------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(v2_signal, color='black', label='Lead V2 Signal', linewidth=1.5)
    
    heat_scaled_v2 = (heatmap / np.max(heatmap)) * (np.nanmax(v2_signal) - np.nanmin(v2_signal)) + np.nanmin(v2_signal)
    plt.plot(heat_scaled_v2, color='red', label='Grad-CAM Attention', alpha=0.7, linewidth=2)
    plt.fill_between(range(TIMESTEPS), np.nanmin(v2_signal), heat_scaled_v2, color='red', alpha=0.2)
    
    plt.title('1D Grad-CAM Saliency Map: Brugada Detection Focus (Lead V2)', fontweight='bold')
    plt.xlabel('Time Steps (10ms / step)')
    plt.ylabel('Amplitude (mV)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show() # This triggers the second popup window


# =====================================================================
# THE MAIN PIPELINE (Putting it all together)
# =====================================================================
def main():
    setup_environment() # 1. Get the computer ready
    
    print("---> Loading Data")
    md, ecg_dict = load_data(FILES_DIR, METADATA_FILE) # 2. Load the patient files
    if not ecg_dict:
        print("No ECG records loaded. Exiting.")
        return
        
    X_array_raw, Y_target = preprocess_and_align(ecg_dict, md) # 3. Clean the heartbeats
    
    print("\n---> Splitting Dataset (80/10/10)")
    # 4. Separate the patients into Train, Practice, and Final Exam groups
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_dataset(X_array_raw, Y_target)
    
    print(f"Training samples:   {len(y_train)} (Brugada: {np.sum(y_train == 1)})")
    print(f"Validation samples: {len(y_val)} (Brugada: {np.sum(y_val == 1)})")
    print(f"Test samples:       {len(y_test)} (Brugada: {np.sum(y_test == 1)})")
    
    print("\n---> Fitting Scalers & Building Tensors")
    # 5. Level out the volumes/sizes of the heartbeats
    means, stds = fit_global_amplitude_scalers(X_train_raw)
    X_train = build_composite_tensor(X_train_raw, means, stds)
    X_val = build_composite_tensor(X_val_raw, means, stds)
    X_test = build_composite_tensor(X_test_raw, means, stds)
    
    class_weights_dict = dict(zip(np.unique(y_train), compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    print("\n---> Training Deep Architecture")
    model = build_clinical_model() # 6. Build the AI Brain
    
    callbacks_list = [
        # If the AI stops improving on the practice tests, stop training early so we don't waste time
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True),
        # If the AI gets stuck, have it take smaller steps (slow down the learning rate)
        callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]
    
    # 7. Start the studying process!
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=8,
              callbacks=callbacks_list, class_weight=class_weights_dict, verbose=1)
              
    print("\n---> Saving Artifacts")
    model.save(MODEL_SAVE_NAME) # Save the brain to our hard drive
    np.savez(SCALERS_SAVE_NAME, means=means, stds=stds)
    print(f"Saved: {MODEL_SAVE_NAME} and {SCALERS_SAVE_NAME}")

    print("\n---> Optimizing Threshold on Validation Set")
    y_val_pred_prob = model.predict(X_val, verbose=0).flatten()
    optimal_thresh = optimize_threshold(y_val, y_val_pred_prob)
    print(f"Best Classification Threshold Found: {optimal_thresh:.2f}")
    
    print("\n---> Evaluating on Unseen Test Set")
    # 8. Take the final exam
    evaluate_and_print_metrics(model, X_test, y_test, optimal_thresh)

    print("\n---> Generating Grad-CAM Interpretability")
    # 9. Draw the picture to show us how the AI made its decision
    visualize_gradcam(model, X_test_raw, y_test, X_test, optimal_thresh, means, stds)


# This just tells Python to start the script at the "main()" function above.
if __name__ == "__main__":
    main()