import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---- Default Parameters ----
I0 = 1.0
DEFAULT_z_max = 50.0
DEFAULT_a0 = 0.05
DEFAULT_b0 = 0.02
DEFAULT_THRESHOLD = 0.05
h = 1.0

# ---------- Attenuation and Simulation ----------
def compute_attenuation_coefficient(wavelength, env_data):
    turbidity = env_data['turbidity']
    λ_opt = 450 + min(turbidity * 10, 100)
    a = a0 + 5e-5 * (wavelength - λ_opt) ** 2
    b = b0 * (turbidity ** 2) * (wavelength / 500.0)
    return a + b

def intensity_derivative(z, I, c):
    return -c * I

def compute_intensity_numerical(wavelength, env_data):
    c = compute_attenuation_coefficient(wavelength, env_data)
    ode_fun = lambda z, I: intensity_derivative(z, I, c)
    sol = solve_ivp(ode_fun, [0, z_max], [I0], method='RK45', t_eval=[z_max])
    return sol.y[0, -1]

def ai_predict_optimal_wavelength(env_data, wavelength_range=(400, 600), num_points=100):
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    intensities = [h * compute_intensity_numerical(λ, env_data) for λ in wavelengths]
    optimal_index = np.argmax(intensities)
    return wavelengths[optimal_index], intensities, wavelengths

def transmit_data(optimal_wavelength, env_data):
    return h * compute_intensity_numerical(optimal_wavelength, env_data)

# ---------- GUI ----------
class SimulationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Underwater Optical Communication Simulator")
        self.geometry("630x580")
        self.create_widgets()

    def create_widgets(self):
        param_frame = ttk.LabelFrame(self, text="Simulation Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        self.z_max_entry = self.add_entry(param_frame, "Propagation Distance (z_max, m):", DEFAULT_z_max, 0)
        self.a0_entry = self.add_entry(param_frame, "Baseline Absorption (a₀, 1/m):", DEFAULT_a0, 1)
        self.b0_entry = self.add_entry(param_frame, "Baseline Scattering (b₀, 1/m):", DEFAULT_b0, 2)

        self.temp_entry = self.add_entry(param_frame, "Temperature (°C):", 20.0, 3)
        self.salin_entry = self.add_entry(param_frame, "Salinity (PSU):", 35.0, 4)
        self.turb_entry = self.add_entry(param_frame, "Turbidity (NTU):", 1.0, 5)

        self.threshold_entry = self.add_entry(param_frame, "Signal Threshold:", DEFAULT_THRESHOLD, 6)

        run_button = ttk.Button(self, text="Start Simulation", command=self.start_simulation)
        run_button.pack(pady=10)

        self.log_widget = scrolledtext.ScrolledText(self, height=15)
        self.log_widget.pack(fill="both", padx=10, pady=5)

    def add_entry(self, parent, label_text, default, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1)
        return entry

    def start_simulation(self):
        global z_max, a0, b0

        try:
            z_max = float(self.z_max_entry.get())
            a0 = float(self.a0_entry.get())
            b0 = float(self.b0_entry.get())
            threshold = float(self.threshold_entry.get())

            env_data = {
                'temperature': float(self.temp_entry.get()),
                'salinity': float(self.salin_entry.get()),
                'turbidity': float(self.turb_entry.get())
            }

        except ValueError:
            self.log_widget.insert(tk.END, "❌ Invalid input. Please enter numeric values.\n")
            return

        self.log_widget.delete(1.0, tk.END)
        self.log_widget.insert(tk.END, f"Environmental Data:\n")
        for k, v in env_data.items():
            self.log_widget.insert(tk.END, f"  {k.capitalize()}: {v:.2f}\n")

        turb = env_data['turbidity']
        λ_opt_theoretical = 450 + min(turb * 10, 100)
        self.log_widget.insert(tk.END, f"  λ_opt (absorption min): {λ_opt_theoretical:.1f} nm\n")

        optimal_wavelength, intensities, wavelengths = ai_predict_optimal_wavelength(env_data)
        received_intensity = transmit_data(optimal_wavelength, env_data)

        self.log_widget.insert(tk.END, f"\nOptimal Wavelength: {optimal_wavelength:.2f} nm\n")
        self.log_widget.insert(tk.END, f"Received Intensity: {received_intensity:.4f}\n")

        if received_intensity >= threshold:
            self.log_widget.insert(tk.END, f"✅ Signal strength sufficient.\n")
        else:
            self.log_widget.insert(tk.END, f"⚠️ Signal strength low. Consider adjusting parameters.\n")

        self.log_widget.see(tk.END)

        # Plot results
        plt.figure()
        plt.plot(wavelengths, intensities, label="Received Intensity")
        plt.axvline(optimal_wavelength, color='red', linestyle='--', label="Optimal λ")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (normalized)")
        plt.title("Intensity vs. Wavelength")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ---------- Launch ----------
if __name__ == '__main__':
    app = SimulationGUI()
    app.mainloop()
