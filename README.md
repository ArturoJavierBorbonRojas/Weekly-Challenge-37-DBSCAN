# WeekLY Challenge 37: Unsupervised Deep Learning (Autoencoder for Anomaly Detection) DBSCAN

## Description
For Week 37, I merged Deep Learning with Unsupervised Anomaly Detection by building a **Neural Network Autoencoder** from scratch using NumPy.

In educational data mining, supervised models are great at classifying known dropout patterns. However, they often fail when confronted with rare, atypical student behaviors (e.g., a top-performing student experiencing sudden burnout and absenteeism). An Autoencoder solves this without requiring labeled anomaly data. It learns to compress normal student profiles into a low-dimensional **Latent Space** and reconstruct them. When fed an anomalous profile, the network fails to reconstruct it accurately, and the spike in Reconstruction Error triggers an early warning alert.

## How it works (The Math)
An Autoencoder consists of two symmetrical architectures connected by a bottleneck:
1. **The Encoder:** Compresses the high-dimensional input $X \in \mathbb{R}^d$ into a lower-dimensional latent representation $Z \in \mathbb{R}^k$ (where $k < d$):
   $$Z = \sigma(X \cdot W_{enc} + b_{enc})$$
2. **The Decoder:** Attempts to reconstruct the original input $\hat{X} \in \mathbb{R}^d$ from the compressed bottleneck $Z$:
   $$\hat{X} = \sigma(Z \cdot W_{dec} + b_{dec})$$
3. **Anomaly Scoring:** The network is trained via Backpropagation to minimize the **Mean Squared Error (MSE)** between the input and its reconstruction:
   $$\mathcal{L}(X, \hat{X}) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2$$
   Any new instance exhibiting a reconstruction error greater than $\mu_{error} + 3\sigma_{error}$ is flagged as a statistical anomaly.

## Technical Highlights
* **Self-Supervised Backpropagation:** The target label of the network is the input itself ($y = X$). The chain rule derivatives propagate seamlessly from the reconstruction layer back through the latent bottleneck to the encoder weights.
* **Statistical Thresholding:** Automatically calculates the $3\sigma$ (three standard deviations) confidence boundary after training to isolate outliers objectively.

## 🛠 Dependencies
* Python 3.14.7
* NumPy
