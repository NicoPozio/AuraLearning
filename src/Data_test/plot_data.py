import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('aura_test_data.csv')

# Subtract the first timestamp so the X-axis starts at 0 seconds
df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0

# ---------------------------------------------------------
# FIGURE 1: Gaze Tracking (Raw vs Smoothed)
# ---------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig1.suptitle('Gaze Tracking: 1 Euro Filter Stabilization', fontsize=16)

ax1.plot(df['timestamp'], df['rawGazeX'] * 1000, label='Raw Webcam Iris X (Scaled)', color='red', alpha=0.3)
ax1.plot(df['timestamp'], df['smoothGazeX'], label='Filtered Screen X', color='blue', linewidth=2)
ax1.set_ylabel('X Coordinate')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(df['timestamp'], df['rawGazeY'] * 1000, label='Raw Webcam Iris Y (Scaled)', color='red', alpha=0.3)
ax2.plot(df['timestamp'], df['smoothGazeY'], label='Filtered Screen Y', color='green', linewidth=2)
ax2.set_ylabel('Y Coordinate')
ax2.set_xlabel('Time (Seconds)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# ---------------------------------------------------------
# FIGURE 2: Comprehensive Emotion & Frustration Analysis
# ---------------------------------------------------------
fig2, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig2.suptitle('UI Metrics Breakdown vs. Total Stress Accumulator', fontsize=16)

axes[0].plot(df['timestamp'], df['zAU4'], label='AU4 (Corrugator) Z-Score', color='purple', linewidth=2)
axes[0].axhline(y=1.5, color='orange', linestyle='--', label='Noise Threshold (1.5σ)')
axes[0].set_ylabel('σ (Std Dev)')
axes[0].set_title('Facial Muscle Tension (Brows)')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(df['timestamp'], df['zEar'], label='EAR (Visual Effort) Z-Score', color='blue', linewidth=2)
axes[1].axhline(y=1.5, color='orange', linestyle='--', label='Noise Threshold (1.5σ)')
axes[1].set_ylabel('σ (Std Dev)')
axes[1].set_title('Visual Effort (Squinting)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.6)

axes[2].plot(df['timestamp'], df['zLip'], label='AU24 (Lip Compression) Z-Score', color='green', linewidth=2)
axes[2].axhline(y=1.5, color='orange', linestyle='--', label='Noise Threshold (1.5σ)')
axes[2].set_ylabel('σ (Std Dev)')
axes[2].set_title('Lip Compression')
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.6)

axes[3].plot(df['timestamp'], df['stressPercentage'], label='Stress Accumulator (%)', color='red', linewidth=2)
axes[3].fill_between(df['timestamp'], 0, df['stressPercentage'], where=(df['isFrustrated']==1), color='red', alpha=0.3, label='Frustration Triggered')
axes[3].axhline(y=100, color='black', linestyle=':', label='100% Capacity')
axes[3].set_ylabel('Fill (%)')
axes[3].set_xlabel('Time (Seconds)')
axes[3].set_title('Total Cognitive Overload (Differential Integration)')
axes[3].legend(loc='upper right')
axes[3].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# ---------------------------------------------------------
# FIGURE 3: Diagnostics & Hardware Stability (NEW)
# ---------------------------------------------------------
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig3.suptitle('System Diagnostics: Hardware Lag & User Distance', fontsize=16)

# 1. Delta Time (dtSec) - Browser Performance
axes3[0].plot(df['timestamp'], df['dtSec'], label='Frame Delta Time (dtSec)', color='orange', linewidth=1.5)
axes3[0].set_ylabel('Seconds')
axes3[0].set_title('Browser Performance & Lag Spikes')
axes3[0].legend(loc='upper right')
axes3[0].grid(True, linestyle='--', alpha=0.6)

# 2. Inter-Ocular Distance (IOD) - Distance Proxy
axes3[1].plot(df['timestamp'], df['iod'], label='Inter-Ocular Distance (IOD)', color='cyan', linewidth=1.5)
axes3[1].set_ylabel('Normalized Width')
axes3[1].set_title('User Distance Proxy (IOD) - Drastic changes break Gaze Tracking')
axes3[1].legend(loc='upper right')
axes3[1].grid(True, linestyle='--', alpha=0.6)

# 3. Stress Delta - Instantaneous Load
axes3[2].plot(df['timestamp'], df['stressDelta'], label='Instantaneous Stress Delta', color='magenta', linewidth=1.5)
axes3[2].axhline(y=0, color='black', linestyle=':', label='Baseline (0)')
axes3[2].set_ylabel('Delta Value')
axes3[2].set_xlabel('Time (Seconds)')
axes3[2].set_title('Instantaneous Load on Stress Accumulator per Frame')
axes3[2].legend(loc='upper right')
axes3[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Show all plots on screen
plt.show()