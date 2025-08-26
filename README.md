# README for Drone Swarm Simulation

## Overview
The Drone Swarm Simulation project is designed to simulate the behavior of a swarm of drones using various algorithms and techniques. The simulation allows for the visualization of drone movements, interactions, and the effects of environmental factors such as obstacles.

Formation Error Reduction

Throughout iterative development, we progressively reduced the swarm’s formation error (measured as root-mean-square error, RMSE) by introducing several algorithmic refinements. Initially, the baseline controller combined only separation, alignment, cohesion, and goal forces, which resulted in RMSE values exceeding 4 m. Subsequent modifications addressed both path efficiency and formation stability: (i) collision handling and potential-field tuning to avoid wall/obstacle penetration, (ii) adaptive formation radius scaling to maintain safe inter-drone spacing, (iii) Hungarian assignment with hysteresis to stabilize slot allocation, (iv) forward-clearance and time-to-collision speed capping to synchronize group velocity, and (v) global path planning via a visibility-graph with Dijkstra search and shortcut smoothing. Finally, low-pass filtering of slot targets, adaptive PD control laws with natural frequency and damping parameters, and a cohesion governor for leader speed regulation significantly tightened slot tracking. Empirically, these steps reduced mean RMSE from >4 m to approximately 1.2 m, with minimum RMSE below 0.5 m, while also lowering settling time and improving overall trajectory efficiency.

Methods

Our approach to swarm formation control and obstacle avoidance was developed incrementally, combining concepts from multi-agent systems, motion planning, and control theory. The main methodological components are as follows:

1)Baseline Swarm Dynamics
We implemented a Boids-inspired controller with separation, alignment, cohesion, and goal-seeking terms. This provided a functional but error-prone formation baseline.

2)Collision Handling and Environment Forces
To prevent unrealistic penetration of walls and obstacles, we incorporated repulsive forces with configurable influence radii, along with a no-penetration projection step after state integration.

3)Formation Geometry and Assignment
The nominal formation was defined as a circular lattice whose radius adapted to the number of agents and the minimum inter-drone distance. Slot allocation was optimized using the Hungarian algorithm with a switch penalty, reducing assignment thrashing.

4)Velocity Regulation
Leader and follower velocities were capped using forward-clearance and time-to-collision (TTC) estimates, ensuring anticipatory slowing near obstacles. A group speed cap based on clearance percentiles maintained cohesion across the formation.

5)Path Planning and Trajectory Smoothing
A visibility-graph planner with Dijkstra search generated obstacle-avoiding waypoints. Post-processing with shortcut smoothing and Pure-Pursuit lookahead improved trajectory efficiency and reduced unnecessary detours.

6)Formation Tracking Enhancements
Followers employed a filtered slot target (low-pass filter) to mitigate jitter. Slot tracking was stabilized using adaptive proportional-derivative (PD) control formulated in natural frequency/damping form. A cohesion governor further constrained leader acceleration when follower lag was detected.

7)Parameter Tuning
Parameters such as separation/align/goal weights, lookahead distance, velocity tracking gain, and percentile-based group speed thresholds were iteratively tuned to balance agility and stability.


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
