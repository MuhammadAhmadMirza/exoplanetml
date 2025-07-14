import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import TimeSeries

# Simulate a sample light curve for an eclipsing binary
np.random.seed(42)
period = 2.0  # days
time = np.linspace(0, 10, 500)
flux = 1 - 0.2 * np.exp(-((time % period - period/2)**2)/(2*0.05**2))  # synthetic eclipse
flux += np.random.normal(0, 0.01, size=flux.shape)  # add noise

# Create an Astropy TimeSeries
ts = TimeSeries(time=time, data={'flux': flux})

# 1. Basic Light Curve Plot
def plot_light_curve(ts):
    plt.figure(figsize=(8, 4))
    plt.scatter(ts.time.value, ts['flux'], s=5, color='black')
    plt.xlabel("Time [days]")
    plt.ylabel("Flux")
    plt.title("Light Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. Phase-Folded Light Curve
def plot_phase_folded(ts, period):
    phase = (ts.time.value % period) / period
    plt.figure(figsize=(8, 4))
    plt.scatter(phase, ts['flux'], s=5, color='blue')
    plt.xlabel("Orbital Phase")
    plt.ylabel("Flux")
    plt.title("Phase-Folded Light Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. Residual Plot
def plot_residuals(ts):
    model_flux = np.ones_like(ts['flux'])  # flat model for residual comparison
    residuals = ts['flux'] - model_flux
    plt.figure(figsize=(8, 4))
    plt.scatter(ts.time.value, residuals, s=5, color='red')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Time [days]")
    plt.ylabel("Residuals (Flux - Model)")
    plt.title("Residuals Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the functions
plot_light_curve(ts)
plot_phase_folded(ts, period)
plot_residuals(ts)
