DroneWarSim

DroneWarSim is a drone swarm simulation framework designed to model and test swarm behaviors in defense-oriented and research scenarios.

The system simulates swarm dynamics, communication losses, sensor errors, and advanced path planning algorithms, making the simulation more realistic for military-style missions, robotics research, and AI development.

✨ Features

Modular Architecture → organized into src/, comms/, planner/, sensors/, ui/, tests/

Communication Model → LossyChannel simulates delay, jitter, and packet loss between drones

Sensor Modeling → GPS noise, dropout, and bias (with placeholders for IMU and LiDAR)

Reporting System → automatic JSON, PNG, and HTML reports after each simulation run

Advanced Planning → scaffolding for A*, D* Lite (dynamic replanning), and RRT*

Dynamic Obstacles → foundation for moving threats and risk-based path planning

Formation Control → Boids-inspired separation, alignment, cohesion, and goal-seeking

Linux/Ubuntu Support → tested on Ubuntu with Python 3.10+

droneSwarmSim/
  main.py                # Simulation entry point
  comms/                 # Communication modules (e.g., LossyChannel)
  planner/               # Path planning (A*, D* Lite, RRT*)
  sensors/               # Sensor models (GPS, IMU, LiDAR)
  reports/               # Auto-generated reports
  raporlama.py           # Reporting script
  tests/                 # Unit and integration tests
⚙️ Requirements

Install the required packages:

pip install numpy matplotlib scipy


Optional (for reporting and advanced features):

pip install pandas


If running on Ubuntu, ensure Python 3.10+ is installed:

sudo apt update
sudo apt install python3 python3-pip

🚀 Running the Simulation

Clone the repository:

git clone https://github.com/<your-username>/DroneWarSim.git
cd DroneWarSim


Run the simulation:

python3 main.py


Results will appear in the terminal and automatically saved under reports/:

timestamp_summary.json → raw results

timestamp_paths.png → path length comparison

timestamp_metrics.png → RMSE & settling time

timestamp_report.html → full HTML report

📊 Example Output
=== Simulation Summary ===
Total path (sum): 2192.99 m
  Drone 0: 269.97 m
  Drone 1: 269.48 m
  ...
Mean RMSE: 1.084 m
Min  RMSE: 0.325 m
Settling time: 26.52 s

🎯 Future Work

Enemy Drones → patrol, interceptor, or jammer agents

Mission System → recon, escort, attack, and delivery tasks

ROS2 Integration → real-time telemetry publishing/subscription

UI Dashboard → live swarm visualization via FastAPI + React

Reinforcement Learning → AI-based dynamic obstacle avoidance
