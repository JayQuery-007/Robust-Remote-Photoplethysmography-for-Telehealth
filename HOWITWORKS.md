# How This System Estimates Vitals

This project has two heart-rate paths:

- A primary classical rPPG pipeline that runs continuously on the webcam ROI.
- An optional deep-learning model that predicts a BVP waveform and can be fused in when enabled.

The Streamlit app uses the classical path by default because it is simpler to inspect and usually more stable in live use.

## 1. Video capture and face ROI selection

The app opens the webcam in [streamlit_app.py](streamlit_app.py) and extracts a face ROI using `FaceROIExtractor` from [equiphys_core.py](equiphys_core.py).

`FaceROIExtractor` works like this:

- It prefers MediaPipe face landmarks when available.
- It uses forehead and cheek landmarks as the pulse region.
- It builds a convex-hull mask over those landmarks.
- It crops that region and resizes it to a fixed size.
- If MediaPipe is unavailable, it falls back to an OpenCV face box with an elliptical face mask.

Why this matters:

- The forehead and upper cheeks are better perfused than the whole face.
- Masking reduces contamination from hair, background, jawline, and other low-signal areas.
- Resizing every ROI to a fixed size removes distance-related amplitude changes.

## 2. Buffered real-time signal formation

For each frame, the app stores:

- The normalized masked ROI for classical rPPG.
- A timestamp for that frame.
- A shorter clip buffer for the optional deep model.

The classical heart-rate path waits for about 10 seconds of data before estimating vitals. The app recomputes roughly once per second using the rolling buffer.

This is important because the pulse is weak in RGB video. A short window is too noisy; a long window is more stable.

## 3. Heart-rate estimation: classical rPPG path

The main HR estimate is produced from the masked ROI buffer using POS and CHROM in [equiphys_core.py](equiphys_core.py).

### Step 3.1: RGB trace extraction

The app spatially averages each ROI frame into one RGB triplet over time. That creates a time series shaped like:

- red(t)
- green(t)
- blue(t)

These traces are the raw optical signals used for rPPG.

### Step 3.2: Illumination normalization

Before pulse extraction, the code applies per-channel normalization with `_illumination_rectify()`.

Purpose:

- suppress slow lighting drift
- reduce channel-scale imbalance
- make pulse information easier to separate from brightness changes

### Step 3.3: Timestamp-based resampling

The code does not assume the camera runs at exactly 30 FPS.

Instead it:

- stores real timestamps for each frame
- computes an effective frame rate from the timestamps
- resamples the RGB traces onto a uniform timeline using cubic splines

Why this matters:

- webcams jitter
- dropped frames distort the frequency axis
- resampling makes spectral estimation physically meaningful

### Step 3.4: POS and CHROM pulse extraction

The current HR pipeline uses POS and CHROM, not raw green averaging, as the main estimators.

POS:

- normalizes the RGB segment by its local mean
- projects color changes onto a plane orthogonal to skin tone
- combines those projected signals into a pulse waveform

CHROM:

- builds chrominance combinations of RGB
- suppresses motion and specular components differently from POS
- produces a second pulse waveform for cross-checking

Why this matters:

- the pulse is not simply "green gets darker"
- specular reflection and motion often dominate raw channel averages
- POS and CHROM are designed to isolate the pulsatile component more directly

### Step 3.5: Bandpass filtering and Welch PSD

After pulse extraction, the code estimates heart rate in the physiological band.

The current live app is configured to search 60 to 120 BPM, which is 1.0 to 2.0 Hz.

Inside `_welch_peak_hz()` the code:

- detrends the signal
- applies a Butterworth bandpass filter
- computes Welch's power spectral density estimate
- finds all candidate peaks with `scipy.signal.find_peaks`
- selects the peak with the highest prominence, not just the tallest bin

Prominence is used because it measures how much a peak stands out above the surrounding noise floor. That is usually a better indicator of a real pulse than absolute amplitude alone.

### Step 3.6: Fusion of classical candidates

The app currently uses POS and CHROM as the main spectral HR candidates.

- If both agree reasonably, the estimate is reinforced.
- If one is noisy and one is clean, the higher-quality candidate wins.
- The optional deep model can also be fused in, but only when enabled and only when it is not wildly inconsistent with the classical estimate.

## 4. Temporal tracking of heart rate

The app does not show the raw spectral HR directly. It applies an SNR-gated EMA in [streamlit_app.py](streamlit_app.py).

Current behavior:

- On the first valid estimate, the display initializes directly from the measured HR.
- If the spectral estimate has high SNR and differs meaningfully from the current tracked value, the display uses a high EMA alpha to snap quickly.
- If the signal is noisy, the EMA alpha is very small so the display barely moves.
- Otherwise it uses a moderate alpha for normal smoothing.

This is the stability layer. Its job is to reduce display flicker without hard-locking the result to a single BPM.

There is also an optional lock mode in the UI. When enabled, the app can hold a stable value once enough recent HR estimates are tightly clustered.

## 5. Optional deep model path

If `Use deep model in fusion` is enabled in the sidebar, the app also runs `EquiPhysDANN` from [equiphys_core.py](equiphys_core.py).

What the model does:

- Input: a video clip tensor shaped like `[batch, channels, time, height, width]`
- Feature extractor: learns a latent representation from the face clip
- Pulse head: predicts a blood-volume-pulse-like waveform over time
- Domain discriminator: predicts nuisance domains such as skin/light categories through a gradient reversal layer

Why the discriminator exists:

- it encourages the latent representation to keep pulse information
- it discourages the latent code from overfitting to lighting or demographic domain cues

The model output is not directly shown as BPM. Instead:

- it predicts a BVP waveform
- BPM is estimated from that waveform separately
- that BPM can be fused with the classical estimate if enabled

There is also optional test-time adaptation in the UI, which updates the model slightly on the live clip before using it.

## 6. SpO2 estimation

SpO2 in this project is an RGB-camera estimate, not a clinical pulse-oximeter replacement.

The code in `estimate_spo2_from_rgb()` works like this:

- average the ROI into RGB traces
- use red and blue channels as webcam proxies for absorption behavior
- separate AC and DC components with bandpass filtering
- compute a ratio-of-ratios
- convert that to an SpO2 estimate with an empirical calibration equation

Important limitation:

- webcams do not have a real infrared channel
- this makes SpO2 estimation approximate and sensitive to camera properties and lighting

So the SpO2 number is best treated as a rough screening signal, not a clinical reading.

## 7. Respiratory-rate estimation

Respiratory rate is estimated from modulation of the pulse waveform in `estimate_respiratory_rate()`.

The code:

- takes a BVP-like signal
- computes its envelope using the Hilbert transform
- bandpass filters that envelope in the breathing band, about 0.15 to 0.4 Hz
- converts the dominant respiratory frequency to breaths per minute

This is based on the idea that breathing modulates pulse amplitude and timing.

## 8. Signal quality and failure cases

The system is most reliable when:

- the face fills a reasonable part of the frame
- lighting is steady and frontal
- the subject is relatively still
- skin regions are not saturated or shadowed

The system becomes less reliable when:

- the face detector falls back to a coarse face box
- there is strong head motion
- auto exposure changes aggressively
- the camera compresses color heavily
- the pulse signal is too weak relative to noise

In those cases the app may hold the previous HR, smooth heavily, or produce lower-confidence estimates.

## 9. In short

Heart rate:

- ROI selection from face landmarks
- masked RGB extraction over time
- timestamp-based resampling
- POS and CHROM pulse extraction
- Welch PSD peak selection by prominence
- SNR-gated temporal smoothing

SpO2:

- red/blue AC/DC ratio heuristic from the ROI RGB traces

Respiratory rate:

- Hilbert-envelope analysis of a BVP-like signal in the respiratory band

Deep model:

- optional clip-to-BVP network with adversarial domain disentanglement

## 10. Files to read next

If you want to inspect the implementation directly, start here:

- [streamlit_app.py](streamlit_app.py): live app loop, buffering, fusion, smoothing, UI
- [equiphys_core.py](equiphys_core.py): ROI extraction, POS/CHROM HR estimation, SpO2, RR, deep model