# HydroCell

A quantum-mechanical simulation framework for modeling electron transport in hydrogen fuel cells.

## Overview

The purpose of this project is to simulate the quantum tunneling of electrons across proton exchange membranes in hydrogen fuel cells. The model combines classical electrochemistry with quantum mechanical effects to provide insights into fuel cell performance at the nanoscale.

## Installation

```bash
git clone https://github.com/zbuhrer/hydrocell.git
cd hydrocell
pip install -r requirements.txt
```

## Project Structure

```
.
├── 01_basic_simulation.ipynb   # Basic simulation examples
├── requirements.txt            # Required dependencies
└── src
    ├── __init__.py
    └── hydrocell.py            # Core simulation code
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

HydroCell models quantum tunneling using the time-independent Schrödinger equation:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

The tunneling probability through a barrier is calculated using:

$$T \approx \exp\left(-2\kappa L\right)$$

where $\kappa = \sqrt{\frac{2m(V_0-E)}{\hbar^2}}$ is the decay constant.

## Features

- Quantum tunneling simulation through potential barriers
- Classical electrochemical calculations for fuel cells
- Interactive visualization of electron wavefunctions
- Parameter sweeps to explore optimal conditions
- Multiple barrier modeling for realistic membrane representation
