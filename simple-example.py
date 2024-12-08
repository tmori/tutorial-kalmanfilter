import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0.0
        self.estimated_error = 1.0

    def update(self, measurement):
        # Prediction update
        self.estimated_error += self.process_variance
        
        # Measurement update
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        self.estimated_value += kalman_gain * (measurement - self.estimated_value)
        self.estimated_error *= (1 - kalman_gain)

        return self.estimated_value

def generate_noisy_data(function, start, stop, num_points, noise_std):
    x = np.linspace(start, stop, num_points)
    y = function(x)
    noise = np.random.normal(0, noise_std, num_points)
    y_noisy = y + noise
    return x, y, y_noisy

def apply_kalman_filter(noisy_data, process_variance, measurement_variance):
    kf = KalmanFilter(process_variance, measurement_variance)
    filtered_data = [kf.update(measurement) for measurement in noisy_data]
    return filtered_data

def main():
    # Step 1: Generate noisy data
    true_function = lambda x: np.sin(x)  # True signal: sin(x)
    x, y_true, y_noisy = generate_noisy_data(true_function, 0, 10, 100, noise_std=0.5)

    # Step 2: Apply Kalman filter
    process_variance = 1e-2
    measurement_variance = 0.5**2
    y_filtered = apply_kalman_filter(y_noisy, process_variance, measurement_variance)

    # Step 3: Save and plot results
    results = pd.DataFrame({"x": x, "True Signal": y_true, "Noisy Signal": y_noisy, "Filtered Signal": y_filtered})
    results.to_csv("filtered_data.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True Signal", linestyle="--")
    plt.plot(x, y_noisy, label="Noisy Signal", alpha=0.5)
    plt.plot(x, y_filtered, label="Filtered Signal")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Kalman Filter on Noisy Signal")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
