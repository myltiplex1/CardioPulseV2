import os
import time
from collections import deque, Counter

import numpy as np
import pandas as pd
import streamlit as st
import wave
import matplotlib.pyplot as plt
import librosa.display
import plotly.graph_objects as go

import model_inference
import cardiac_parameters
import audio_enhancement
from esp32_streams import AudioReceiver, MotionReceiver
from spectrogram_utils import TARGET_SR, EXPECTED_SAMPLES, audio_to_logmel, FMAX

# --- CONFIGURATION ---
ESP32_IP = "192.168.4.1"
TCP_PORT_AUDIO = 3333
TCP_PORT_MOTION = 3334
CLASSIFICATION_STEP_SEC = 2
CARDIAC_PARAMS_STEP_SEC = 15
CARDIAC_PARAMS_BUFFER_SEC = 15
VOTE_WINDOW_SIZE = 5
BPM_HISTORY_SIZE = 3
BPM_OUTLIER_FRACTION = 0.4
SAVE_DIR = "saved_recordings"
CLASSES = ["Normal", "Murmur", "Extra_Systole", "Artifact"]

st.set_page_config(page_title="CardioPulse", layout="wide")

# --- Visual styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 5rem !important; font-weight: 900 !important; margin-bottom: 0 !important;
        line-height: 1.1 !important;
    }
    .sub-header {
        font-size: 2.2rem !important; opacity: 0.8 !important; margin-top: 0 !important;
    }
    .status-badge-connected {
        background-color: #064e3b; color: #6ee7b7; padding: 4px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }
    .status-badge-disconnected {
        background-color: #7f1d1d; color: #fca5a5; padding: 4px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }
    .status-badge-offline {
        background-color: #78350f; color: #fcd34d; padding: 4px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }
    .class-card {
        padding: 24px; border-radius: 16px; text-align: center; margin-bottom: 10px;
    }
    .class-card-green {
        background-color: #064e3b; border: 2px solid #10b981; color: #6ee7b7 !important;
    }
    .class-card-orange {
        background-color: #78350f; border: 2px solid #f59e0b; color: #fcd34d !important;
    }
    .class-card-green .class-label, .class-card-green div { color: #6ee7b7 !important; }
    .class-card-orange .class-label, .class-card-orange div { color: #fcd34d !important; }
    .class-label { font-size: 1.8rem; font-weight: 700; }

    .section-card {
        background-color: #1a1d24; border-radius: 12px; padding: 16px; border: 1px solid #2d3139;
        color: #e5e7eb !important;
    }
    .section-card *, .section-card p, .section-card strong { color: #e5e7eb !important; }
    .tier-experimental { opacity: 0.9; }

    .param-grid {
        display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px;
    }
    .param-box {
        background-color: #12151c; border: 1px solid #2d3139; border-radius: 10px;
        padding: 12px 16px; min-width: 140px; flex: 1;
    }
    .param-box-label {
        font-size: 0.75rem; color: #9ca3af !important; text-transform: uppercase;
        letter-spacing: 0.03em; margin-bottom: 4px;
    }
    .param-box-value {
        font-size: 1.4rem; font-weight: 700; color: #f3f4f6 !important;
    }
</style>
""", unsafe_allow_html=True)


def render_param_boxes(items):
    """Renders a row of boxed parameter cards from a list of (label, value) pairs."""
    boxes = "".join(
        f'<div class="param-box"><div class="param-box-label">{label}</div>'
        f'<div class="param-box-value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="param-grid">{boxes}</div>'


def style_dark_axes(fig, ax):
    """Applies dark-theme styling to a matplotlib figure/axes so charts
    match the surrounding Streamlit dark theme, instead of matplotlib's
    default white background."""
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='#9ca3af', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2d3139')
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    ax.title.set_color('#e5e7eb')
    ax.grid(alpha=0.15, color='#6b7280')




def display_class_html(smoothed_class, agreement):
    if smoothed_class == "Uncertain":
        return '<div class="class-card class-card-orange"><div class="class-label">Still gathering a clear signal...</div></div>'
    card_class = "class-card-green" if agreement >= 0.6 else "class-card-orange"
    return f'<div class="class-card {card_class}"><div class="class-label">{smoothed_class}</div><div>{agreement:.0%} agreement (last {VOTE_WINDOW_SIZE} readings)</div></div>'


def smooth_bpm(new_bpm, bpm_history):
    """Median-smooths BPM across recent readings, with outlier rejection."""
    if new_bpm is None:
        return (bpm_history[-1] if bpm_history else None), bpm_history

    if bpm_history:
        current_smoothed = float(np.median(bpm_history))
        if current_smoothed > 0 and abs(new_bpm - current_smoothed) / current_smoothed > BPM_OUTLIER_FRACTION:
            return current_smoothed, bpm_history

    bpm_history.append(new_bpm)
    if len(bpm_history) > BPM_HISTORY_SIZE:
        bpm_history.pop(0)
    return float(np.median(bpm_history)), bpm_history


@st.cache_resource
def load_models_cached():
    model_inference.load_models()
    return True


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if "audio_receiver" not in st.session_state:
        st.session_state.audio_receiver = None
        st.session_state.motion_receiver = None
        st.session_state.connected = False
        st.session_state.running = False

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<p class="main-header">CardioPulse</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Hi Enoch</p>', unsafe_allow_html=True)
    with header_col2:
        is_connected = st.session_state.connected
        badge_class = "status-badge-connected" if is_connected else "status-badge-disconnected"
        badge_text = "CardioPulse connected" if is_connected else "Not connected"
        st.markdown(f'<div style="text-align:right; padding-top:20px;">'
                    f'<span class="{badge_class}">{badge_text}</span></div>',
                    unsafe_allow_html=True)

    load_models_cached()

    col_connect, col_start, col_stop, _ = st.columns([1, 1, 1, 3])
    if col_connect.button("Connect to CardioPulse device", disabled=st.session_state.connected):
        try:
            st.session_state.audio_receiver = AudioReceiver(ESP32_IP, TCP_PORT_AUDIO)
            st.session_state.motion_receiver = MotionReceiver(ESP32_IP, TCP_PORT_MOTION)
            st.session_state.audio_receiver.start()
            st.session_state.motion_receiver.start()
            st.session_state.connected = True
            st.rerun()
        except Exception as e:
            st.error(f"Connection failed: {e}")

    if col_start.button("Start monitoring", disabled=st.session_state.running or not st.session_state.connected, type="primary"):
        st.session_state.audio_receiver.start_recording()
        st.session_state.motion_receiver.mark_recording_start(time.time())
        st.session_state.running = True
    if col_stop.button("Stop", disabled=not st.session_state.running):
        st.session_state.audio_receiver.stop_recording()
        st.session_state.running = False

    status_placeholder = st.empty()
    st.divider()

    # --- Classification section ---
    class_col, prob_col = st.columns([1, 1])
    with class_col:
        class_placeholder = st.empty()
        raw_class_placeholder = st.empty()
    with prob_col:
        st.markdown("**Class probabilities (this reading)**")
        prob_chart_placeholder = st.empty()

    st.divider()

    # --- Motion section (live) ---
    st.markdown("### Motion")
    motion_col1, motion_col2 = st.columns(2)
    with motion_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Reading reliability** (motion-artifact flagging)")
        st.caption("Flags whether motion occurred during the specific window "
                   "a cardiac reading came from -- tells you whether to trust "
                   "that particular BPM/HRV reading.")
        reliability_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    with motion_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Activity context**")
        st.caption("A longer rolling average (resting vs. active) for interpreting "
                   "readings -- e.g. an elevated BPM means something different "
                   "at rest than right after walking.")
        activity_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- Cardiac parameters ---
    st.markdown("### Cardiac Parameters")
    val_col, exp_col = st.columns(2)
    with val_col:
        st.markdown("**Validated**")
        validated_boxes_placeholder = st.empty()
    with exp_col:
        st.markdown("**Experimental** (not yet validated against ground truth)")
        experimental_boxes_placeholder = st.empty()

    st.divider()

    # --- Trend charts ---
    st.markdown("### Trends")
    st.caption("BPM (smoothed)")
    bpm_chart_placeholder = st.empty()
    st.caption("Quality Score (signal reliability, not a health trend)")
    quality_chart_placeholder = st.empty()

    with st.expander("Experimental trends (HRV, rhythm regularity -- not yet validated)"):
        st.caption("HRV (SDNN, ms)")
        hrv_chart_placeholder = st.empty()
        st.caption("Rhythm Regularity")
        reg_chart_placeholder = st.empty()

    st.divider()

    # --- Waveform, envelope, spectrogram ---
    st.markdown("### Signal View")
    sig_col1, sig_col2 = st.columns(2)
    with sig_col1:
        st.caption("Waveform (raw vs. filtered) -- last classification window")
        waveform_placeholder = st.empty()
    with sig_col2:
        st.caption("Smoothed PCG Envelope -- last cardiac analysis window")
        envelope_placeholder = st.empty()

    st.caption("Beat Detection (alternating intervals, approximate S1/S2)")
    st.caption("Note: dashed lines mark detected beats, colored by alternation -- "
              "not a validated S1-vs-S2 identification, see the systolic/diastolic "
              "ratio limitation discussed earlier.")
    beat_detection_placeholder = st.empty()

    st.caption("Mel-Spectrogram -- what the model sees")
    spectrogram_placeholder = st.empty()

    st.divider()

    # --- Playback ---
    st.markdown("### Listen")
    st.caption("Raw / filtered / noise-reduced versions of the last cardiac analysis window")
    playback_placeholder = st.empty()

    if not st.session_state.connected:
        status_placeholder.info("Click 'Connect to CardioPulse device' to begin.")
        return
    if not st.session_state.running:
        status_placeholder.info("Connected. Click 'Start monitoring' to begin live analysis.")
        return

    audio = st.session_state.audio_receiver
    motion = st.session_state.motion_receiver

    last_classification_time = 0
    last_cardiac_time = 0
    vote_history = deque(maxlen=VOTE_WINDOW_SIZE)
    bpm_smoothing_history = []
    latest_cardiac = None
    bpm_history, quality_history, hrv_history, reg_history = [], [], [], []

    while st.session_state.running:
        now = time.time()

        recent = audio.get_recent_samples(EXPECTED_SAMPLES)
        if recent is None:
            status_placeholder.text("Waiting for enough audio to begin classifying...")
            time.sleep(0.3)
            continue

        if now - last_classification_time >= CLASSIFICATION_STEP_SEC:
            raw_class, raw_confidence, probs = model_inference.classify_chunk(recent)
            vote_history.append(raw_class)
            last_classification_time = now
        else:
            time.sleep(0.1)
            continue

        vote_counts = Counter(vote_history)
        smoothed_class, smoothed_count = vote_counts.most_common(1)[0]
        agreement = smoothed_count / len(vote_history)

        # --- Waveform for this classification window ---
        chunk = np.array(recent, dtype=np.int16)
        y_raw_chunk = chunk.astype(np.float32) / 32768.0
        y_filtered_chunk = model_inference.preprocess_chunk(chunk)

        t = np.linspace(0, len(y_raw_chunk) / TARGET_SR, num=len(y_raw_chunk))

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=y_raw_chunk, mode='lines', name='Raw',
                                      line=dict(color='#60a5fa', width=1), opacity=0.6))
        fig_wave.add_trace(go.Scatter(x=t, y=y_filtered_chunk, mode='lines', name='Filtered',
                                      line=dict(color='#f472b6', width=1)))
        fig_wave.update_layout(
            template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            title='Raw Acoustic Waveform (current window)', title_font_size=14,
            xaxis_title='Time (s)', margin=dict(l=40, r=20, t=40, b=30), height=280,
            legend=dict(x=0.85, y=1.1, bgcolor='rgba(0,0,0,0)'),
        )
        waveform_placeholder.plotly_chart(fig_wave, use_container_width=True)

        # --- Spectrogram for this classification window ---
        spec = audio_to_logmel(y_filtered_chunk, TARGET_SR)
        fig_spec, ax2 = plt.subplots(figsize=(10, 2.5))
        librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=TARGET_SR, fmax=FMAX, ax=ax2, cmap='magma')
        ax2.set_title('Mel-Spectrogram (model input)', fontsize=10)
        style_dark_axes(fig_spec, ax2)
        fig_spec.tight_layout()
        spectrogram_placeholder.pyplot(fig_spec)
        plt.close(fig_spec)

        # --- Cardiac parameters + motion, every CARDIAC_PARAMS_STEP_SEC ---
        if now - last_cardiac_time >= CARDIAC_PARAMS_STEP_SEC:
            n_samples_needed = int(CARDIAC_PARAMS_BUFFER_SEC * TARGET_SR)
            recent_long = audio.get_recent_samples(n_samples_needed)

            if recent_long is not None:
                window_start = now - CARDIAC_PARAMS_BUFFER_SEC
                long_chunk = np.array(recent_long, dtype=np.int16)
                tmp_path = os.path.join(SAVE_DIR, "_tmp_dashboard_live.wav")
                with wave.open(tmp_path, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(TARGET_SR)
                    wf.writeframes(long_chunk.tobytes())
                latest_cardiac = cardiac_parameters.extract_cardiac_parameters(tmp_path)

                smoothed_bpm, bpm_smoothing_history = smooth_bpm(latest_cardiac["bpm"], bpm_smoothing_history)
                latest_cardiac["bpm_raw"] = latest_cardiac["bpm"]
                latest_cardiac["bpm"] = smoothed_bpm

                if latest_cardiac["bpm"]:
                    bpm_history.append(latest_cardiac["bpm"])
                    quality_history.append(latest_cardiac["quality_score"])
                if latest_cardiac.get("hrv_sdnn_ms"):
                    hrv_history.append(latest_cardiac["hrv_sdnn_ms"])
                if latest_cardiac.get("rhythm_regularity_cv"):
                    reg_history.append(latest_cardiac["rhythm_regularity_cv"])

                # --- Motion-artifact flagging for THIS cardiac window ---
                window_motion_scores = motion.get_scores_in_window(window_start, now)
                if window_motion_scores:
                    high_frac = sum(1 for s in window_motion_scores if s >= 0.10) / len(window_motion_scores)
                    if high_frac > 0.10:
                        reliability_text = "LOW -- motion detected during this reading"
                        reliability_color = "#fcd34d"
                    elif high_frac > 0:
                        reliability_text = "Moderate -- some motion detected"
                        reliability_color = "#fcd34d"
                    else:
                        reliability_text = "Good -- minimal motion during capture"
                        reliability_color = "#6ee7b7"
                else:
                    reliability_text = "Unknown"
                    reliability_color = "#9ca3af"
                reliability_placeholder.markdown(
                    f'<div style="font-size:1.3rem; font-weight:700; color:{reliability_color};">{reliability_text}</div>',
                    unsafe_allow_html=True)

                # --- Envelope + beat detection ---
                envelope = latest_cardiac.get("envelope")
                peak_indices = latest_cardiac.get("peak_indices")
                if envelope is not None:
                    t_env = np.linspace(0, len(envelope) / TARGET_SR, num=len(envelope))

                    fig_env = go.Figure()
                    fig_env.add_trace(go.Scatter(
                        x=t_env, y=envelope, mode='lines', line=dict(color='#34d399', width=1.5),
                        fill='tozeroy', fillcolor='rgba(52, 211, 153, 0.3)', name='Envelope'
                    ))
                    fig_env.update_layout(
                        template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                        title='Smoothed PCG Envelope', title_font_size=14,
                        xaxis_title='Time (s)', margin=dict(l=40, r=20, t=40, b=30), height=280,
                        showlegend=False,
                    )
                    envelope_placeholder.plotly_chart(fig_env, use_container_width=True)

                    fig_beat = go.Figure()
                    fig_beat.add_trace(go.Scatter(
                        x=t_env, y=envelope, mode='lines', line=dict(color='#e5e7eb', width=1),
                        name='Envelope'
                    ))
                    if peak_indices is not None and len(peak_indices) > 0:
                        for i, p in enumerate(peak_indices):
                            color = '#f87171' if i % 2 == 0 else '#60a5fa'
                            fig_beat.add_vline(x=p / TARGET_SR, line_dash='dash', line_color=color, line_width=1.5)
                    fig_beat.update_layout(
                        template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                        title='Beat Detection (alternating S1/S2 approx.)', title_font_size=14,
                        xaxis_title='Time (s)', margin=dict(l=40, r=20, t=40, b=30), height=280,
                        showlegend=False,
                    )
                    beat_detection_placeholder.plotly_chart(fig_beat, use_container_width=True)

                # --- Playback versions ---
                versions = audio_enhancement.get_playback_versions(long_chunk.tolist())
                with playback_placeholder.container():
                    pcol1, pcol2, pcol3 = st.columns(3)
                    pcol1.caption("Raw")
                    pcol1.audio((versions["raw"] * 32767).astype(np.int16), sample_rate=TARGET_SR)
                    pcol2.caption("Filtered (model input)")
                    pcol2.audio((versions["filtered"] * 32767).astype(np.int16), sample_rate=TARGET_SR)
                    pcol3.caption("Enhanced (noise-reduced)")
                    pcol3.audio((versions["enhanced"] * 32767).astype(np.int16), sample_rate=TARGET_SR)

            last_cardiac_time = now

        # --- Activity context (independent of cardiac cycle -- always current) ---
        activity_avg, activity_label = motion.get_activity_context()
        activity_color = "#6ee7b7" if activity_label == "resting" else "#fcd34d" if activity_label == "active" else "#9ca3af"
        activity_extra = f" (avg score: {activity_avg:.3f})" if activity_avg is not None else ""
        activity_placeholder.markdown(
            f'<div style="font-size:1.3rem; font-weight:700; color:{activity_color};">'
            f'{activity_label.capitalize()}{activity_extra}</div>', unsafe_allow_html=True)

        # --- Update remaining displays ---
        status_placeholder.text(f"Monitoring live -- last update: {time.strftime('%H:%M:%S')}")

        class_placeholder.markdown(display_class_html(smoothed_class, agreement), unsafe_allow_html=True)
        raw_class_placeholder.caption(f"Raw (this single reading): {raw_class} ({raw_confidence:.2f} confidence)")

        prob_values = [probs.get(c, 0) for c in CLASSES]
        prob_df = pd.DataFrame({"Probability": prob_values}, index=CLASSES)
        prob_chart_placeholder.bar_chart(prob_df, height=200)

        if latest_cardiac:
            bpm_val = f"{latest_cardiac['bpm']:.1f}" if latest_cardiac['bpm'] else "--"
            if latest_cardiac.get('bpm_raw') and latest_cardiac['bpm']:
                bpm_val += f" <span style='font-size:0.7rem; opacity:0.6;'>(raw: {latest_cardiac['bpm_raw']:.1f})</span>"

            validated_items = [
                ("BPM (smoothed)", bpm_val),
                ("BPM Confidence", latest_cardiac.get('bpm_confidence', '--')),
                ("Systolic/Diastolic", f"{latest_cardiac['systolic_diastolic_ratio']:.3f}"
                 if latest_cardiac['systolic_diastolic_ratio'] else "--"),
                ("Quality Score", f"{latest_cardiac['quality_score']:.0f}/100"
                 if latest_cardiac['quality_score'] else "--"),
                ("Peaks Detected", str(latest_cardiac.get('n_peaks_detected', '--'))),
            ]
            validated_boxes_placeholder.markdown(render_param_boxes(validated_items), unsafe_allow_html=True)

            experimental_items = [
                ("HRV (SDNN)", f"{latest_cardiac['hrv_sdnn_ms']:.1f}ms"
                 if latest_cardiac['hrv_sdnn_ms'] else "--"),
                ("S1/S2 Amplitude Ratio", f"{latest_cardiac['s1_s2_amplitude_ratio']:.2f}"
                 if latest_cardiac['s1_s2_amplitude_ratio'] else "--"),
                ("Rhythm Regularity", f"{latest_cardiac['rhythm_regularity_cv']:.3f}"
                 if latest_cardiac['rhythm_regularity_cv'] else "--"),
            ]
            experimental_boxes_placeholder.markdown(render_param_boxes(experimental_items), unsafe_allow_html=True)

        if len(bpm_history) >= 2:
            bpm_chart_placeholder.line_chart({"BPM": bpm_history}, height=150)
            quality_chart_placeholder.line_chart({"Quality": quality_history}, height=150)
        if len(hrv_history) >= 2:
            hrv_chart_placeholder.line_chart({"HRV": hrv_history}, height=150)
        if len(reg_history) >= 2:
            reg_chart_placeholder.line_chart({"Regularity": reg_history}, height=150)

        time.sleep(0.3)

    status_placeholder.success("Monitoring stopped.")


if __name__ == "__main__":
    main()
