import numpy as np
from scipy import constants
from scipy.integrate import solve_bvp


F = 96485  # Faraday constant (C/mol)
R = 8.314  # Gas constant (J/mol·K)
ΔH = -285830  # Enthalpy (J/mol)
ΔG = -237130  # Gibbs free energy (J/mol)

class FuelCell:
    def __init__(self, T=298.15, PH2=1.0, PO2=1.0, membrane_resistance=0.05,
                 barrier_width=1.0, barrier_height=4.0, electron_energy=2.5):
        # Classical parameters
        self.T = T
        self.PH2 = PH2
        self.PO2 = PO2
        self.membrane_resistance = membrane_resistance

        # Quantum parameters
        self.barrier_width = barrier_width  # nm
        self.barrier_height = barrier_height  # eV
        self.electron_energy = electron_energy  # eV

        # Constants for quantum calculations
        self.hbar = constants.hbar  # J·s
        self.m_e = constants.m_e  # kg
        self.e = constants.e  # C

        # Convert to SI units for calculations
        self.barrier_width_m = self.barrier_width * 1e-9  # convert nm to m
        self.barrier_height_J = self.barrier_height * self.e  # convert eV to J
        self.electron_energy_J = self.electron_energy * self.e  # convert eV to J

    def open_circuit_voltage(self):
        E0 = 1.229  # V at standard conditions
        return E0 + (R * self.T / (2 * F)) * np.log(self.PH2 * np.sqrt(self.PO2))

    def potential_barrier(self, x):
        """Define the potential barrier function"""
        # Simple rectangular barrier
        if 0 <= x <= self.barrier_width_m:
            return self.barrier_height_J
        return 0.0

    def solve_schrodinger(self):
        """Solve the time-independent Schrödinger equation for a barrier"""
        # Set up the computational domain
        x_range = np.linspace(-3*self.barrier_width_m, 4*self.barrier_width_m, 1000)

        # Wave vector k
        k = np.sqrt(2 * self.m_e * self.electron_energy_J) / self.hbar

        # Define the Schrödinger equation as a first-order system
        def schrodinger_system(x, y):
            psi, phi = y
            V = np.array([self.potential_barrier(xi) for xi in x])
            dpsi_dx = phi
            dphi_dx = 2 * self.m_e / (self.hbar**2) * (V - self.electron_energy_J) * psi
            return np.vstack((dpsi_dx, dphi_dx))

        # Boundary conditions: incident wave from left and transmitted wave to right
        def bc(ya, yb):
            # Left boundary: psi = exp(ikx) + R*exp(-ikx)
            # Right boundary: psi = T*exp(ikx)
            return np.array([
                ya[0] - (1 + 0.5),  # Normalization at left boundary
                yb[0] - 0.5         # Arbitrary value at right boundary
            ])

        # Initial guess for the solution
        y_guess = np.zeros((2, len(x_range)))
        y_guess[0] = np.exp(-((x_range - x_range.mean())**2)/2)  # Gaussian guess
        y_guess[1] = np.gradient(y_guess[0], x_range)

        # Solve the boundary value problem
        try:
            sol = solve_bvp(schrodinger_system, bc, x_range, y_guess, max_nodes=10000)
            return sol.x, sol.y[0]  # x values and wavefunction psi
        except:
            # Fallback to analytical approximation if numerical solution fails
            return self.wkb_approximation(x_range)

    def wkb_approximation(self, x_range):
        """WKB approximation for tunneling through a rectangular barrier"""
        # Wave vector inside and outside the barrier
        k_outside = np.sqrt(2 * self.m_e * self.electron_energy_J) / self.hbar

        if self.electron_energy_J < self.barrier_height_J:
            kappa = np.sqrt(2 * self.m_e * (self.barrier_height_J - self.electron_energy_J)) / self.hbar
            # Simple wavefunctions for demonstration
            psi = np.zeros_like(x_range, dtype=complex)
            for i, x in enumerate(x_range):
                if x < 0:
                    psi[i] = np.exp(1j * k_outside * x) + 0.5 * np.exp(-1j * k_outside * x)
                elif 0 <= x <= self.barrier_width_m:
                    psi[i] = 0.5 * np.exp(-kappa * x)
                else:
                    psi[i] = 0.25 * np.exp(1j * k_outside * (x - self.barrier_width_m))
            return x_range, np.abs(psi)
        else:
            # Energy above barrier
            k_inside = np.sqrt(2 * self.m_e * (self.electron_energy_J - self.barrier_height_J)) / self.hbar
            psi = np.zeros_like(x_range, dtype=complex)
            for i, x in enumerate(x_range):
                if x < 0:
                    psi[i] = np.exp(1j * k_outside * x) + 0.2 * np.exp(-1j * k_outside * x)
                elif 0 <= x <= self.barrier_width_m:
                    psi[i] = 0.8 * np.exp(1j * k_inside * x)
                else:
                    psi[i] = 0.7 * np.exp(1j * k_outside * (x - self.barrier_width_m))
            return x_range, np.abs(psi)

    def compute_tunneling_probability(self):
        """Calculate the tunneling probability through the barrier"""
        if self.electron_energy_J >= self.barrier_height_J:
            # Above-barrier transmission
            return 1.0
        else:
            # Tunneling probability using WKB approximation
            kappa = np.sqrt(2 * self.m_e * (self.barrier_height_J - self.electron_energy_J)) / self.hbar
            return np.exp(-2 * kappa * self.barrier_width_m)

    def quantum_enhanced_current(self, classical_current):
        """Adjust the classical current based on quantum tunneling effects"""
        tunneling_prob = self.compute_tunneling_probability()

        # Scale the current by the tunneling probability
        # This is a simplified model; in reality, the relationship would be more complex
        quantum_factor = 0.1 + 0.9 * tunneling_prob  # Ensures some current even with low tunneling

        return classical_current * quantum_factor

    def simulate(self, load_resistance):
        V_oc = self.open_circuit_voltage()
        R_total = self.membrane_resistance + load_resistance

        # Classical current calculation
        classical_current = V_oc / R_total

        # Apply quantum effects
        current = self.quantum_enhanced_current(classical_current)

        voltage = current * load_resistance
        power = voltage * current
        efficiency = (-ΔG / ΔH) * (voltage / V_oc)

        # Include quantum metrics in the result
        return {
            "voltage": voltage,
            "current": current,
            "power": power,
            "efficiency": efficiency,
            "V_oc": V_oc,
            "tunneling_probability": self.compute_tunneling_probability(),
            "quantum_factor": current / classical_current if classical_current > 0 else 0
        }
