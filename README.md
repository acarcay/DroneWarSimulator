# README for Drone Swarm Simulation

## Overview
The Drone Swarm Simulation project is designed to simulate the behavior of a swarm of drones using various algorithms and techniques. The simulation allows for the visualization of drone movements, interactions, and the effects of environmental factors such as obstacles.

## Project Structure
The project consists of the following files:

- **config.py**: Contains the `Config` class for simulation configuration parameters.
- **control.py**: Defines the `SwarmController` class to manage drone swarm behavior.
- **environment.py**: Contains the `Environment` class for calculating repulsion forces from walls and obstacles.
- **main.py**: The entry point for the simulation, initializing all components and starting the animation loop.
- **metrics.py**: Defines the `Metrics` class for tracking swarm performance metrics.
- **models.py**: Contains the `Drone` class representing individual drones in the simulation.
- **utils.py**: Provides utility functions for vector operations and calculations.
- **viz.py**: Defines the `Visualizer` class for handling the graphical representation of the simulation.
- **README.md**: Documentation for the project.

## Requirements
To run the simulation, ensure you have the following Python packages installed:

- numpy
- matplotlib
- scipy (optional, for optimal assignment)

You can install the required packages using pip:

```
pip install numpy matplotlib scipy
```

## Running the Simulation
1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Run the `main.py` file:

```
python main.py
```

4. Use the following controls during the simulation:
   - Press the spacebar to pause/resume the simulation.
   - Press 'g' to toggle between waypoint and obstacle modes.
   - Use 'tab' and 'backspace' to cycle through obstacles.

## Customization
You can modify the simulation parameters in the `config.py` file to adjust the behavior of the drones, the environment, and the visualization settings.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.