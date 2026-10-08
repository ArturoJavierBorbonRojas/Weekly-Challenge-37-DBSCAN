import numpy as np

# Week 37: Unsupervised Deep Learning - Autoencoder for Anomaly Detection
# Author: Ing. Arturo Javier Borbon Rojas

print("Initializing Deep Autoencoder (Anomaly Detection Engine)...")
print("-" * 65)

class Autoencoder:
    def __init__(self, input_dim, latent_dim, learning_rate=0.1):
        np.random.seed(42)
        # Encoder weights (Compression: 4D -> 2D)
        self.W_enc = np.random.uniform(-0.5, 0.5, (input_dim, latent_dim))
        self.b_enc = np.zeros((1, latent_dim))
        
        # Decoder weights (Reconstruction: 2D -> 4D)
        self.W_dec = np.random.uniform(-0.5, 0.5, (latent_dim, input_dim))
        self.b_dec = np.zeros((1, input_dim))
        
        self.lr = learning_rate
        self.threshold = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, epochs=5000):
        print(f"Compressing {X.shape[1]}D data into a {self.W_enc.shape[1]}D Latent Space over {epochs} epochs:")
        
        for epoch in range(epochs):
            # 1. FORWARD PASS
            # Encode (Compress to bottleneck)
            latent_input = np.dot(X, self.W_enc) + self.b_enc
            latent_space = self.sigmoid(latent_input)
            
            # Decode (Reconstruct original input)
            reconstructed_input = np.dot(latent_space, self.W_dec) + self.b_dec
            reconstructed_X = self.sigmoid(reconstructed_input)
            
            # 2. RECONSTRUCTION ERROR (X - X_hat)
            error = X - reconstructed_X
            
            # 3. BACKPROPAGATION (Chain Rule across Decoder -> Encoder)
            d_reconstructed = error * self.sigmoid_derivative(reconstructed_X)
            
            error_latent = np.dot(d_reconstructed, self.W_dec.T)
            d_latent = error_latent * self.sigmoid_derivative(latent_space)
            
            # 4. GRADIENT DESCENT UPDATES
            self.W_dec += np.dot(latent_space.T, d_reconstructed) * self.lr
            self.b_dec += np.sum(d_reconstructed, axis=0, keepdims=True) * self.lr
            
            self.W_enc += np.dot(X.T, d_latent) * self.lr
            self.b_enc += np.sum(d_latent, axis=0, keepdims=True) * self.lr
            
            if epoch % 1000 == 0 or epoch == epochs - 1:
                mse = np.mean(np.square(error))
                print(f"   Epoch {epoch:04d} | Global Reconstruction Loss (MSE): {mse:.6f}")

        # Set anomaly threshold Automatically (Mean Error + 3 Standard Deviations)
        train_errors = self.get_reconstruction_error(X)
        self.threshold = np.mean(train_errors) + 3 * np.std(train_errors)
        print(f"\n Calibration Complete. Anomaly MSE Threshold set at: {self.threshold:.6f}")

    def reconstruct(self, X):
        latent = self.sigmoid(np.dot(X, self.W_enc) + self.b_enc)
        return self.sigmoid(np.dot(latent, self.W_dec) + self.b_dec)

    def get_reconstruction_error(self, X):
        """Calculates the Mean Squared Error (MSE) per student profile."""
        X_hat = self.reconstruct(X)
        return np.mean(np.square(X - X_hat), axis=1)

# --- Normalized Educational Dataset (0.0 to 1.0) ---
# Features: [Grades, Attendance_Rate, Assignment_Completion, SocioEmotional_Stability]
# Training ONLY on "Normal/Regular" coherent student patterns
X_normal_train = np.array([
    [0.90, 0.95, 0.92, 0.88], [0.85, 0.90, 0.85, 0.85], [0.95, 0.98, 0.96, 0.92], # High performers
    [0.75, 0.80, 0.78, 0.75], [0.70, 0.75, 0.72, 0.70], [0.80, 0.85, 0.80, 0.80], # Average regulars
    [0.65, 0.70, 0.68, 0.65], [0.88, 0.92, 0.90, 0.86], [0.72, 0.78, 0.74, 0.72]  # Regular patterns
])

# Initialize Autoencoder: 4 Input Features -> 2 Latent Neurons -> 4 Reconstructed Outputs
autoencoder = Autoencoder(input_dim=4, latent_dim=2, learning_rate=0.2)
autoencoder.fit(X_normal_train, epochs=5000)

print("-" * 65)
print("EVALUATING NEW PROFILES FOR HIDDEN ANOMALIES:")

X_test = np.array([
    [0.82, 0.86, 0.84, 0.81], # Profile 1: Typical regular student
    [0.98, 0.20, 0.95, 0.10], # Profile 2: ANOMALY (Straight-A student suddenly skipping class & severe emotional drop)
    [0.15, 0.95, 0.10, 0.90]  # Profile 3: ANOMALY (Perfect attendance & emotionally stable, but failing everything)
])

errors = autoencoder.get_reconstruction_error(X_test)
reconstructions = autoencoder.reconstruct(X_test)

for i, (orig, rec, err) in enumerate(zip(X_test, reconstructions, errors)):
    is_anomaly = err > autoencoder.threshold
    status = "ANOMALY DETECTED (Atypical Risk)" if is_anomaly else "Normal Pattern"
    print(f"\nStudent {i+1}:")
    print(f"   Original Input:      {orig.round(2)}")
    print(f"   NN Reconstruction:   {rec.round(2)}")
    print(f"   Reconstruction MSE:  {err:.6f} ➔ {status}")