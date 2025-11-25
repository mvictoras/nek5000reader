# Nek5000 Reader

A Python library for reading and processing Nek5000 simulation files.
Supports both serial and MPI-parallel reading.

## Installation

```bash
pip install .
```

## Usage

```python
from mpi4py import MPI
from nek5000reader import Nek5000Reader

# Initialize reader
reader = Nek5000Reader("simulation.nek5000", MPI.COMM_WORLD)

# Read a timestep
data = reader.read_timestep(0)

# Access data
coords = data['coordinates']
velocity = data['fields']['velocity']
```

## Testing

```bash
pip install .[test]
pytest
```
