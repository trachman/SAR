import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, fftconvolve

#
# Example of Pulse Compression (AKA Matched Filtering)
#

# GOAL
# I want to understand why we send a Linear FM Signal and how it simulates higher resolution
# This was explained by taking the convolution of two known signals and comparing their overlap

# Parameters
#
Fs = 5000 # Sample rate (Hz)
T = 0.02  # Chirp duration (seconds)
f0 = 100  # Start frequency (Hz)
f1 = 1000 # End frequency (Hz)

# Transmit signal: Linear FM Chirp
t = np.arange(0, T, 1 / Fs)
tx = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# Received signal: Delayed + Noise
delay = 300  # samples delay
# rx = np.zeros(len(t) + delay)
rx = np.zeros((len(t) * 2) + delay)
rx[delay : delay + len(tx)] = tx

# Add a second target, see how close we can get with two distinct spikes in the returned convolution of our signal
# Notice how the spikes model a sinc function. This is awesome because that spike is essentially our returned target resolution.
# Our resolution is then how close we can make these spikes without them overlapping.
tx_index = 0
second_target_delay = 310 # We can get our second target response echo to start as close as 10 samples away and still differentiate the peaks!
for i in range(second_target_delay, second_target_delay + len(tx)):
    rx[i] = rx[i] + tx[tx_index]
    tx_index += 1

# Since we can detect targets as little as 10 samples away from each other in delay
# This corresponds to a range resolution of (1 / 5000) * 10 = 0.002 seconds in range delay
# Obviously the range here depends on distance away from the illuminated scene but still cool!

# Add noise
rx += 0.5 * np.random.randn(len(rx))

# Matched filter: Time-Reverse & Conjugate
# Adding windowing to improve the PSLR and make target detection easier (at the cost of some resolution (broader main lobe))
window = np.kaiser(len(tx), 2.5)
tx_with_windowing = tx * window
mf = np.conj(tx_with_windowing[::-1])
y = fftconvolve(rx, mf, mode='same') # Equivalent to np.convolve, just using a different algorithm and exploiting convolution properties in frequency domain
# y = np.convolve(rx, mf, mode='same')

# Find the index of the maximum value of 'y' which is our convolution result
# This will tell us the exact index that the received echo from a target matches up with
# with the transmitted pulse. In our case the delay of the first target is known: 300 samples.
#
# Chirp Duration = 0.02 seconds
#
# At a sample rate of 5000 Hz (5000 samples per second)
# Our sample interval is 1 / 5000 = 0.0002 seconds
#
# 0.02 / 0.0002 = 100
# 
# Which means there is a 100 samples that will make up our chirp and received echo.
# Note: We are assuming no phase shift here
#
# If our received echo is to line up with our chirp, they will be highly correlated when
# their midpoints line up. Ie. this is 100 / 2 = 50 samples into the start of the received echo.
#
# If we have a delay of 300 samples, this means the highest correlation should occur at index
# 300 + 50 of our convolution result.
#
max_index = np.argmax(y)
print(f'Maximum Index of Convolution: {max_index}')
print(f'Estimated Target Delay: {max_index - (T / (1 / Fs) / 2)}')

# Notes:
# All things holding constant
# - The more samples you have, the easier it is to do target detection
# - The shorter the chirp, the easier it is to do target detection (resolution goes up, but at the cost of power)
# - The greater the bandwidth of the chirp, the easier it is to do target detection (pulse compression is better)

# Plot
plt.figure(figsize=(10,7))

plt.subplot(3,1,1)
plt.plot(t, tx)
plt.title("Transmitted Chirp Signal (No Windowing)")
plt.ylabel("Amplitude")

plt.subplot(3,1,2)
plt.plot(rx)
plt.title("Received Signal (with delay + noise)")
plt.ylabel("Amplitude")

plt.subplot(3,1,3)
plt.plot(y)
plt.title("Matched Filter Output (Pulse Compression)")
plt.ylabel("Amplitude")
plt.xlabel("Sample Index")

plt.tight_layout()
plt.show()

