# HydroCell

A quantum-mechanical simulation framework for modeling electron transport in hydrogen fuel cells.

## Overview

The purpose of this project is to simulate the quantum tunneling of electrons across proton exchange membranes (PEMs) in hydrogen fuel cells. The model combines classical electrochemistry with quantum mechanical effects to provide insights into fuel cell performance at the nanoscale.

## Installation

```bash
git clone https://github.com/zbuhrer/hydrocell.git
cd hydrocell
pip install -r requirements.txt
```

## Usage

```python
from src.hydrocell import FuelCell

# Create a fuel cell with custom parameters
cell = FuelCell(
    T=298.15,                # Temperature (K)
    PH2=1.0,                 # Hydrogen pressure (atm)
    PO2=1.0,                 # Oxygen pressure (atm)
    membrane_resistance=0.05, # Membrane resistance (Ω)
    barrier_width=0.5,        # Barrier width (nm)
    barrier_height=3.0,       # Barrier height (eV)
    electron_energy=2.7       # Electron energy (eV)
)

# Run simulation with a specific load resistance
result = cell.simulate(load_resistance=0.3)

# Display results
for k, v in result.items():
    print(f"{k.capitalize()}: {v:.3f}")
```

## Mathematical Foundation

HydroCell models quantum tunneling using the time-independent Schrödinger equation for an electron traversing a potential barrier $V(x)$:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

where $\psi$ is the electron wavefunction, $m$ is the electron mass, $E$ is the electron energy, and $\hbar$ is the reduced Planck constant.

For a simple rectangular barrier of height $V_0$ and width $L$ (where $V_0 > E$), the tunneling probability $T$ is approximately given by the WKB approximation:

$$T \approx \exp\left(-2\kappa L\right)$$

where $\kappa = \sqrt{\frac{2m(V_0-E)}{\hbar^2}}$ is the decay constant within the barrier. This formula highlights the strong dependence of tunneling on barrier width ($L$), barrier height ($V_0$), and electron energy ($E$).

## Realistic Membrane Modeling

The `EnhancedFuelCell` class extends the basic model to incorporate more realistic physical phenomena:

1.  **Complex Barrier Shapes:** The `potential_barrier(self, x)` method can model different spatially varying potential profiles:
    *   `rectangular`: The basic model.
    *   `gaussian`: A smooth, bell-shaped barrier.
    *   `double_well`: A barrier with dips, potentially representing intermediate states or structural features.
    *   `nafion`: A simplified representation of a common PEM material, including periodic potential fluctuations due to features like sulfonate groups.

2.  **Hydration Level:** For the `nafion` barrier, the `hydration_level` parameter ($\lambda$, on a scale of 0 to 1) influences the potential profile. Higher hydration levels typically lower the energy barriers or deepen potential wells within the membrane, facilitating charge transport (in this model, via affecting the electron potential). The depth of the periodic wells in the simplified Nafion model increases linearly with $\lambda$.

3.  **Temperature Effects:** Temperature ($T$) affects the system in several ways. In this enhanced model, temperature is included in the calculation of the *effective* tunneling probability via phenomenological factors applied to the base tunneling probability. These factors crudely represent effects like thermal broadening of electron energy distribution and potential thermal activation aspects of the barrier.

4.  **Quantum Enhanced Current:** The overall current output of the fuel cell is modeled as a base classical current scaled by a "quantum factor". This factor combines the calculated tunneling probability (including temperature effects) and a hydration-dependent conductivity term (for the Nafion-like model):

    $$I_{quantum} = I_{classical} \times F_{quantum}$$
    $$F_{quantum} = \alpha + (1-\alpha) \times T_{temp-dep} \times F_{hydration}$$
    where $\alpha$ is a base conductivity factor (e.g., 0.1), $T_{temp-dep}$ is the temperature-dependent tunneling probability, and $F_{hydration}$ is a hydration factor (e.g., $0.5 + 0.5 \times \lambda$). Note that this is a simplified model for combining effects, not a rigorous derivation from first principles quantum transport.

## Features

- Quantum tunneling simulation using the time-independent Schrödinger equation.
- **Support for multiple realistic potential barrier shapes (rectangular, gaussian, double well, nafion).**
- **Modeling of hydration level influence on the barrier profile (for nafion).**
- **Inclusion of simplified temperature effects on tunneling probability.**
- Classical electrochemical calculations for fuel cells.
- **Calculation of quantum-enhanced current using a combined model.**
- Interactive visualization of electron wavefunctions and potential profiles.
- Parameter sweeps to explore optimal conditions and the influence of quantum effects.
