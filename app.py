import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import tensorflow as tf
import wfdb
import tempfile
import os
from datetime import datetime


# =====================================================================
# GLOBAL SETTINGS (The App's Control Panel)
# =====================================================================
FS = 100  # "Sample Rate": 100 snapshots of the heartbeat every second.
TIMESTEPS = 1200  # We look at exactly 1200 snapshots (12 seconds) total.
NUM_LEADS = 12  # The number of sensors placed on the patient's body.
CUTOFF_FREQ = 0.5  # Used for our digital "noise-cancellation" filter.
MODEL_PATH = "brugada_champion_model.keras"  # Where the AI's brain is saved.
SCALER_PATH = "brugada_scalers.npz"  # Where the AI's volume-leveling rules are saved.

# The EXACT order the AI was trained on. Do not change this!
STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Set up the look and feel of the web page
st.set_page_config(page_title="Brugada ECG Classifier", page_icon="🫀", layout="wide")


# =====================================================================
# SYSTEM PREPARATION
# =====================================================================
@st.cache_resource
def load_champion_model():
    """Loads the AI's brain into memory ONCE so the app stays fast."""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# =====================================================================
# DATA CLEANING & PREPARATION (Translating for the AI)
# =====================================================================
def preprocess_ecg(df: pd.DataFrame) -> np.ndarray:
    """Translates raw hospital data into the mathematical format the AI understands."""
    if not os.path.exists(SCALER_PATH):
        st.error(f"⚠️ Scaler file `{SCALER_PATH}` not found. Please ensure it is in the app directory.")
        st.stop()

    scalers = np.load(SCALER_PATH)
    means, stds = scalers['means'], scalers['stds']

    # 1. Noise Cancellation (High-pass Filter)
    b, a = butter(N=3, Wn=CUTOFF_FREQ / (FS / 2), btype="high")
    
    # Safely lock the columns into the exact order the AI expects
    raw_signals = np.zeros((TIMESTEPS, NUM_LEADS))
    for j, lead_name in enumerate(STANDARD_LEADS):
        if lead_name in df.columns:
            # Grab exactly 1200 timesteps for this specific lead
            signal = df[lead_name].values
            if len(signal) >= TIMESTEPS:
                raw_signals[:, j] = signal[:TIMESTEPS]
            else:
                # Pad with zeros if it's too short
                raw_signals[:len(signal), j] = signal

    # Apply the high-pass filter
    filtered = np.zeros_like(raw_signals)
    for i in range(NUM_LEADS):
        filtered[:, i] = filtfilt(b, a, raw_signals[:, i])

    # 2. Building the "3-Layer Cake" (Tensor Engineering)
    X_norm = np.copy(filtered)
    for j in range(NUM_LEADS):
        lead = X_norm[:, j]
        # Standard deviation fix to safely handle completely dead sensors (flatlines)
        lead_std = np.std(lead)
        safe_std = lead_std if lead_std > 1e-8 else 1.0 
        X_norm[:, j] = (lead - np.mean(lead)) / safe_std

    X_amp = (filtered - means.reshape(1, -1)) / stds.reshape(1, -1)
    mask = np.ones((TIMESTEPS, NUM_LEADS))

    composite = np.concatenate([X_norm, X_amp, mask], axis=-1)
    return np.expand_dims(composite, axis=0).astype(np.float32)


# =====================================================================
# FILE HANDLING (Reading the Doctor's Uploads)
# =====================================================================
def parse_wfdb_uploads(uploaded_files):
    """Glues the .hea and .dat medical files back together."""
    if len(uploaded_files) != 2:
        return None, "Please upload exactly TWO files: one .hea and one .dat", None

    exts = [os.path.splitext(f.name)[1].lower() for f in uploaded_files]
    if '.hea' not in exts or '.dat' not in exts:
        return None, "Missing either the .hea or .dat file.", None

    base_names = set([os.path.splitext(f.name)[0] for f in uploaded_files])
    if len(base_names) > 1:
        return None, "The .hea and .dat files must have the exact same file name.", None

    base_name = list(base_names)[0]

    with tempfile.TemporaryDirectory() as tmpdirname:
        for f in uploaded_files:
            file_path = os.path.join(tmpdirname, f.name)
            with open(file_path, "wb") as out_file:
                out_file.write(f.read())

        record_path = os.path.join(tmpdirname, base_name)
        try:
            record = wfdb.rdrecord(record_path)
            df = pd.DataFrame(record.p_signal, columns=record.sig_name)
            # Strip hidden spaces from column names just in case!
            df.columns = df.columns.str.strip()
            df['Time_s'] = np.arange(len(df)) / record.fs
            return df, None, base_name
        except Exception as e:
            return None, f"WFDB reading error: {e}", None


# =====================================================================
# EXPLAINABLE AI (Grad-CAM Visualizations)
# =====================================================================
def get_last_conv_layer_name(model):
    """Automatically hunts through the AI's brain to find the exact layer needed for mapping."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None


def generate_gradcam_1d(model, model_input_numpy, last_conv_layer_name):
    """Generates a 1D temporal heatmap showing where the model was looking."""
    try:
        model_out = model.output[0] if isinstance(model.output, list) else model.output
        
        # Grab the targeted layer (e.g., conv_block_1 for higher resolution)
        target_layer = model.get_layer(last_conv_layer_name)
        layer_out = target_layer.output[0] if isinstance(target_layer.output, list) else target_layer.output

        grad_model = tf.keras.models.Model(
            inputs=model.inputs, 
            outputs=[layer_out, model_out]
        )

        x_input_tensor = tf.convert_to_tensor(model_input_numpy, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_input_tensor)
            outputs = grad_model(x_input_tensor)
            
            conv_outputs = outputs[0]
            predictions = outputs[1]
            
            if isinstance(conv_outputs, list): conv_outputs = conv_outputs[0]
            if isinstance(predictions, list): predictions = predictions[0]

            loss = predictions[0, 0]

        grads = tape.gradient(loss, conv_outputs)
        if isinstance(grads, list): grads = grads[0]

        pooled_grads = tf.reduce_mean(grads, axis=1)

        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[0]
        
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap.numpy()

        if len(heatmap) == 0:
            return np.zeros(TIMESTEPS)
            
        max_heat = np.max(heatmap)
        if max_heat > 0:
            heatmap /= max_heat

        # Smoothly interpolate the high-res feature map back to the full 1200 timesteps
        f = interp1d(np.arange(len(heatmap)), heatmap, kind='linear', fill_value="extrapolate")
        heatmap_resized = f(np.linspace(0, len(heatmap) - 1, TIMESTEPS))
        
        return heatmap_resized

    except Exception as e:
        st.warning(f"⚠️ Could not generate visual map. Error: {e}")
        return None


# =====================================================================
# VISUALIZATION & REPORTING (Drawing the results)
# =====================================================================
def plot_12_lead_ecg(df: pd.DataFrame):
    """Draws a beautiful grid showing the raw readouts of all 12 sensors."""
    if 'Time_s' not in df.columns:
        df['Time_s'] = np.arange(len(df)) / FS

    ecg_long = df.melt(id_vars=["Time_s"], var_name="Lead", value_name="Amplitude")

    g = sns.FacetGrid(ecg_long, col="Lead", col_wrap=3, sharey=False, height=2, aspect=2.5)
    g.map_dataframe(sns.lineplot, x="Time_s", y="Amplitude", color="#b22222", linewidth=0.8)

    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle("Patient 12-Lead ECG", fontweight='bold', fontsize=14)
    st.pyplot(g.figure)


def plot_high_risk_leads_with_heatmap(raw_df: pd.DataFrame, heatmap: np.ndarray = None):
    """Overlays the AI's attention heatmap behind the raw ECG trace."""
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    temp_df = raw_df.iloc[:TIMESTEPS].copy()
    if 'Time_s' not in temp_df.columns:
        temp_df['Time_s'] = np.arange(len(temp_df)) / FS

    time_data = temp_df['Time_s'].values

    # Setup colors for heatmap (Clear -> Yellow -> Red)
    has_valid_heatmap = heatmap is not None and np.max(heatmap) > 0

    if has_valid_heatmap:
        import matplotlib.colors as mcolors
        colors = [(1, 1, 1, 0), (1, 1, 0, 0.4), (1, 0, 0, 0.6)]
        cm = mcolors.LinearSegmentedColormap.from_list('attention_map', colors, N=100)

    for i, lead in enumerate(['V1', 'V2']):
        if lead in temp_df.columns:
            lead_signal = temp_df[lead].values
            
            # Plot the raw ECG line
            ax[i].plot(time_data, lead_signal, color='black', lw=1.5, zorder=2)

            # Paint the background with the AI's heatmap IF it successfully generated
            if has_valid_heatmap:
                extent = [time_data[0], time_data[-1], ax[i].get_ylim()[0], ax[i].get_ylim()[1]]
                ax[i].imshow(heatmap[np.newaxis, :], cmap=cm, aspect='auto', extent=extent, zorder=1)
                ax[i].set_title(f"Diagnostic Focus: Lead {lead} (Red = Maximum AI Activation)")
            else:
                ax[i].set_title(f"Standard View: Lead {lead} (Heatmap Unavailable)")

            ax[i].set_xlabel("Time (Seconds)")
            ax[i].set_ylabel("Amplitude (mV)")
            ax[i].grid(True, linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    st.pyplot(fig)


def generate_report(is_brugada, probability, patient_id):
    """Automatically types up a medical summary of what the AI found."""
    status = "POSITIVE" if is_brugada else "NEGATIVE"
    current_date = datetime.now().strftime('%Y-%m-%d')

    # DYNAMIC RECOMMENDATIONS BASED ON DIAGNOSIS RESULT
    if is_brugada:
        recommendations = (
            "- Correlate with clinical symptoms (e.g., syncope, family history).\n"
            "- Review the visualizer for the classic 'Coved' ST-elevation in leads V1-V2.\n"
            "- Consider consultation with a cardiac electrophysiologist."
        )
    else:
        recommendations = (
            "- No high-risk Brugada morphology detected in this trace.\n"
            "- Continue standard clinical monitoring if symptoms persist.\n"
            "- Consider serial ECGs if clinical suspicion remains high."
        )

    report = f"""
=======================================================
BRUGADA SYNDROME DIAGNOSTIC ASSISTANT REPORT
=======================================================
Patient ID: {patient_id}
Assessment Date: {current_date}

DIAGNOSIS SUMMARY:
------------------
Result: {status} for Brugada Pattern
Model Confidence: {probability * 100:.2f}%

RECOMMENDATIONS:
----------------
{recommendations}

MODEL SPECIFICATIONS:
---------------------
The hybrid AI architecture analyzed a 36-channel composite tensor 
derived from the 12-lead ECG. The analysis integrated three distinct 
feature layers (Morphology, Amplitude, and Signal Quality) for 
{TIMESTEPS} timesteps. Focus was maintained on the right precordial 
leads (V1, V2) to detect ST-segment elevation and T-wave inversion patterns.

Disclaimer: This AI-generated report is a diagnostic 
aid and should be reviewed by a qualified cardiologist.
=======================================================
"""
    return report


# =====================================================================
# USER INTERFACE MANAGERS (Organizing the Webpage)
# =====================================================================
def handle_file_uploads():
    """Handles the sidebar logic where users drop their files."""
    st.sidebar.header("Patient Input")
    upload_type = st.sidebar.radio("Select Input Format:", ["WFDB (.hea & .dat)", "CSV (.csv)"])

    raw_df = None
    patient_id = "Unknown"

    if upload_type == "WFDB (.hea & .dat)":
        st.sidebar.markdown("Highlight and drag **both** files in at the same time.")
        uploaded_files = st.sidebar.file_uploader("Upload WFDB Files", type=["hea", "dat"], accept_multiple_files=True)

        if uploaded_files:
            raw_df, error_msg, patient_id = parse_wfdb_uploads(uploaded_files)
            if error_msg:
                st.sidebar.error(error_msg)
                st.stop()

    elif upload_type == "CSV (.csv)":
        uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            # Strip column names just in case!
            raw_df.columns = raw_df.columns.str.strip()
            patient_id = os.path.splitext(uploaded_file.name)[0]

    return raw_df, patient_id


def show_diagnosis_dashboard(is_brugada, probability, patient_id):
    """Displays the final Yes/No answer and the confidence bar."""
    st.markdown("---")
    st.header(f"Diagnosis Results (Patient: {patient_id})")
    col1, col2 = st.columns(2)

    with col1:
        if is_brugada:
            st.error("🚨 **POSITIVE for Brugada Pattern**")
            st.write("The model detected waveform features consistent with Brugada Syndrome.")
        else:
            st.success("✅ **NEGATIVE for Brugada Pattern**")
            st.write("The model did not find high-risk indicators in this ECG trace.")

    with col2:
        st.metric(label="Model Confidence", value=f"{probability * 100:.2f}%")
        st.progress(float(probability))
        st.caption(
            "Low Risk <---------------------------------------------|---------------------------------------------> High Risk")


# =====================================================================
# THE MAIN APP LOOP (The Orchestrator)
# =====================================================================
def main():
    # 1. Say Hello
    st.title("🫀 Brugada Syndrome Diagnostic Assistant")
    st.markdown("Upload a patient's 12-lead ECG data to get an automated risk assessment.")

    # 2. Wake up the AI
    try:
        model = load_champion_model()
    except Exception:
        st.error(f"⚠️ Could not load model at `{MODEL_PATH}`. Please ensure the model file exists.")
        st.stop()

    # 3. Wait for the user to upload a file
    raw_df, patient_id = handle_file_uploads()

    # 4. If a file is successfully uploaded, process it!
    if raw_df is not None:

        # Check if the file is too short to be analyzed
        if raw_df.shape[0] < TIMESTEPS:
            st.error(f"Invalid signal length: {raw_df.shape[0]}. Expected at least {TIMESTEPS}.")
            st.stop()

        st.success(f"ECG Data Loaded Successfully for Patient: **{patient_id}**")

        # Give the doctor an option to look at the raw data
        with st.expander("🔍 View Full 12-Lead ECG Trace", expanded=False):
            plot_12_lead_ecg(raw_df.iloc[:TIMESTEPS].copy())

        # 5. Let the AI do the math
        with st.spinner("Analyzing signals..."):
            model_input = preprocess_ecg(raw_df)
            probability = model.predict(model_input, verbose=0)[0][0]
            is_brugada = probability > 0.5

        # 6. Show the big results
        show_diagnosis_dashboard(is_brugada, probability, patient_id)

        # 7. If positive, show the dangerous areas (Explainable AI / Grad-CAM)
        if is_brugada:
            st.info("### 🔍 Visualisation of the Part where it suggests the Brugada")

            with st.spinner("Generating High-Resolution Explainable AI Heatmaps..."):
                # Check for the specific early layer for higher temporal resolution.
                # If it's not found in the architecture, fallback to the last conv layer automatically.
                try:
                    model.get_layer('conv_block_1')
                    layer_name = 'conv_block_1'
                except ValueError:
                    layer_name = get_last_conv_layer_name(model)

                if layer_name:
                    heatmap = generate_gradcam_1d(model, model_input, layer_name)
                    plot_high_risk_leads_with_heatmap(raw_df, heatmap)
                else:
                    st.warning("Could not auto-detect a Convolutional Layer for the heatmap. Showing standard view.")
                    plot_high_risk_leads_with_heatmap(raw_df, None)

        # 8. Let the doctor download the official paperwork
        st.markdown("---")
        st.subheader("📋 Action Items")
        report_text = generate_report(is_brugada, probability, patient_id)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        download_filename = f"Brugada_Report_{patient_id}_{timestamp}.txt"

        st.download_button(
            label="📥 Download Clinical Report",
            data=report_text,
            file_name=download_filename,
            mime="text/plain",
        )

    else:
        # If no file is uploaded yet, just show a friendly prompt
        st.info("👈 Please upload the patient's ECG files in the sidebar to begin.")


if __name__ == "__main__":
    main()