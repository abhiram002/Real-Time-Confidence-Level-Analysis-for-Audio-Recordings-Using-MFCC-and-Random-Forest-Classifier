import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
import threading
import matplotlib.pyplot as plt

# Assuming you have the paths to your "confident" and "non confident" audio folders
confident_folder = r"Voicedata\\confident"
non_confident_folder = r"Voicedata\\Non-confident"

# Extract features from the audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, sr=None)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Take the mean of MFCCs over time
    return mfccs_processed

# Load and process the audio files from the folders
confident_features = np.array([extract_features(f"{confident_folder}/{file}") for file in os.listdir(confident_folder)])
non_confident_features = np.array([extract_features(f"{non_confident_folder}/{file}") for file in os.listdir(non_confident_folder)])

# Create labels for the audio files
confident_labels = np.ones(confident_features.shape[0])
non_confident_labels = np.zeros(non_confident_features.shape[0])

# Combine features and labels
X = np.vstack((confident_features, non_confident_features))
y = np.hstack((confident_labels, non_confident_labels))

# Train a Random Forest classifier using cross-validation
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
accuracies = []
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f"Mean cross-validation accuracy: {np.mean(accuracies):.3f}")

# Real-time prediction
def predict_confidence_level(features):
    confidence_level = clf.predict_proba([features])[0][1] * 10  # Scale the output to 0-10
    return confidence_level

# Start and stop recording functions
def start_recording():
    global recording, audio_data
    recording = True
    audio_data = np.array([])  # Initialize audio_data as an empty NumPy array
    threading.Thread(target=record_audio).start()

def stop_recording():
    global recording, audio_data
    recording = False

# Record audio function
def record_audio():
    global recording, audio_data
    while recording:
        audio = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype="int16")
        sd.wait()
        audio_data = np.concatenate((audio_data, audio.ravel()))  # Update this line

# Main loop
recording = False
audio_data = np.array([])
while True:
    action = input("Press 's' to start recording, 'q' to quit: ")
    if action == 's':
        start_recording()
    elif action == 'q':
        stop_recording()
        break

# Save the recorded audio to a file
file_path = "recorded_audio.wav"
sf.write(file_path, audio_data, 44100)

# Process the recorded audio for confidence level
if len(audio_data) > 0:
    features = extract_features(file_path)
    confidence_level = predict_confidence_level(features)
    print("Confidence level (out of 10):", confidence_level)

    # Plot the audio waveform
    plt.figure(figsize=(12, 4))
    time_axis = np.linspace(0, len(audio_data) / 44100, num=len(audio_data))
    plt.plot(time_axis, audio_data, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Recorded Audio Waveform')
    plt.show()

    # Plot the confidence level graph
    plt.figure(figsize=(6, 4))
    plt.plot([confidence_level], color='r', marker='o')
    plt.xticks([0], [''])
    plt.xlabel('-')
    plt.ylabel('Confidence Level')
    plt.title('Confidence Level')
    plt.show()
else:
    print("No audio data recorded.")