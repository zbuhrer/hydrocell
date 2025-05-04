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
        """solver for the time-independent Schrödinger equation"""
        # Use logarithmically spaced points to better resolve the barrier region
        left_domain = np.linspace(-5*self.barrier_width_m, -0.1*self.barrier_width_m, 200)
        barrier_domain = np.linspace(-0.1*self.barrier_width_m, 1.1*self.barrier_width_m, 400)
        right_domain = np.linspace(1.1*self.barrier_width_m, 5*self.barrier_width_m, 200)
        x_range = np.concatenate([left_domain, barrier_domain, right_domain])
        x_range = np.sort(np.unique(x_range))  # Ensure no duplicate points

        # Wave vector k outside barrier
        k = np.sqrt(2 * self.m_e * self.electron_energy_J) / self.hbar

        # Handle cases where energy > barrier vs energy < barrier
        if self.electron_energy_J >= self.barrier_height_J:
            # For E > V, use the analytical solution directly since there's no tunneling
            # Just reflection/transmission at a potential step
            return self.analytical_above_barrier(x_range)
        else:
            # For E < V, use a combination of analytical and numerical approaches
            try:
                # Define the Schrödinger equation as a first-order system
                def schrodinger_system(x, y):
                    psi, phi = y
                    V = np.zeros_like(x)
                    for i, xi in enumerate(x):
                        V[i] = self.potential_barrier(xi)

                    # Avoid division by zero by adding a small epsilon
                    epsilon = 1e-10
                    dpsi_dx = phi
                    dphi_dx = 2 * self.m_e / (self.hbar**2 + epsilon) * (V - self.electron_energy_J) * psi
                    return np.vstack((dpsi_dx, dphi_dx))

                # Improved boundary conditions based on physical understanding
                def bc(ya, yb):
                    # Left side: incident + reflected wave
                    # Right side: transmitted wave only
                    k_outside = np.sqrt(2 * self.m_e * self.electron_energy_J) / self.hbar

                    # Enforce continuity and derivative matching
                    return np.array([
                        # At left boundary: psi is normalized
                        ya[0] - 1.0,
                        # At right boundary: derivative matches transmitted wave
                        yb[1] - 1j * k_outside * yb[0]
                    ])

                # More robust initial guess
                y_guess = np.zeros((2, len(x_range)))

                # For x < 0
                mask_left = x_range < 0
                y_guess[0, mask_left] = np.exp(1j * k * x_range[mask_left]) + 0.5 * np.exp(-1j * k * x_range[mask_left])
                y_guess[1, mask_left] = 1j * k * np.exp(1j * k * x_range[mask_left]) - 0.5 * 1j * k * np.exp(-1j * k * x_range[mask_left])

                # For 0 <= x <= barrier_width
                kappa = np.sqrt(2 * self.m_e * abs(self.barrier_height_J - self.electron_energy_J)) / self.hbar
                mask_barrier = (x_range >= 0) & (x_range <= self.barrier_width_m)
                y_guess[0, mask_barrier] = 0.5 * np.exp(-kappa * x_range[mask_barrier])
                y_guess[1, mask_barrier] = -0.5 * kappa * np.exp(-kappa * x_range[mask_barrier])

                # For x > barrier_width
                mask_right = x_range > self.barrier_width_m
                y_guess[0, mask_right] = 0.25 * np.exp(1j * k * (x_range[mask_right] - self.barrier_width_m))
                y_guess[1, mask_right] = 0.25 * 1j * k * np.exp(1j * k * (x_range[mask_right] - self.barrier_width_m))

                # Use complex solver for accurate phases
                sol = solve_bvp(schrodinger_system, bc, x_range, y_guess.real,
                               max_nodes=10000, tol=1e-3, verbose=0)

                if not sol.success:
                    # Fall back to analytical approximation if numerical solution fails
                    return self.analytical_tunneling(x_range)

                return sol.x, sol.y[0]

            except Exception as e:
                print(f"Numerical solver failed: {str(e)}")
                # Fallback to analytical approximation
                return self.analytical_tunneling(x_range)

    def analytical_tunneling(self, x_range):
        """Analytical approximation for tunneling wavefunction"""
        # Wave vectors
        k_outside = np.sqrt(2 * self.m_e * self.electron_energy_J) / self.hbar
        kappa = np.sqrt(2 * self.m_e * (self.barrier_height_J - self.electron_energy_J)) / self.hbar

        # Transmission and reflection coefficients (approximate)
        T = 4 * k_outside * kappa / ((k_outside + kappa)**2) * np.exp(-kappa * self.barrier_width_m)
        R = 1 - T  # Conservation of probability

        # Construct wavefunction
        psi = np.zeros_like(x_range, dtype=complex)
        for i, x in enumerate(x_range):
            if x < 0:
                # Incident + reflected wave
                psi[i] = np.exp(1j * k_outside * x) + np.sqrt(R) * np.exp(-1j * k_outside * x)
            elif 0 <= x <= self.barrier_width_m:
                # Evanescent wave in barrier
                psi[i] = (1 + np.sqrt(R)) * np.exp(-kappa * x)
            else:
                # Transmitted wave
                psi[i] = np.sqrt(T) * np.exp(1j * k_outside * (x - self.barrier_width_m))

        return x_range, psi

    def analytical_above_barrier(self, x_range):
        """Analytical solution for electron energy above barrier height"""
        # Wave vectors
        k_outside = np.sqrt(2 * self.m_e * self.electron_energy_J) / self.hbar
        k_inside = np.sqrt(2 * self.m_e * (self.electron_energy_J - self.barrier_height_J)) / self.hbar

        # Reflection and transmission at each interface
        r1 = (k_outside - k_inside) / (k_outside + k_inside)
        t1 = 2 * k_outside / (k_outside + k_inside)

        r2 = (k_inside - k_outside) / (k_inside + k_outside)
        t2 = 2 * k_inside / (k_inside + k_outside)

        # Phase accumulated across barrier
        phase = k_inside * self.barrier_width_m

        # Total transmission coefficient
        T = abs(t1 * t2 * np.exp(1j * phase) / (1 + r1 * r2 * np.exp(2j * phase)))**2
        R = 1 - T

        # Construct wavefunction
        psi = np.zeros_like(x_range, dtype=complex)
        for i, x in enumerate(x_range):
            if x < 0:
                # Incident + reflected wave
                psi[i] = np.exp(1j * k_outside * x) + np.sqrt(R) * np.exp(-1j * k_outside * x)
            elif 0 <= x <= self.barrier_width_m:
                # Oscillatory wave in barrier
                # Approximate behavior - real solution would need to match at boundaries
                psi[i] = t1 * (np.exp(1j * k_inside * x) + r2 * np.exp(1j * k_inside * (2 * self.barrier_width_m - x)))
            else:
                # Transmitted wave
                psi[i] = np.sqrt(T) * np.exp(1j * k_outside * x)

        return x_range, psi

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
