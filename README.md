# PyConDE & PyData 2025 - Pi_Optimal Sprint: Reward Function Engineering

Welcome to the hands-on coding session at PyConDE & PyData 2025! This sprint focuses on applying Pi_Optimal for supermarket inventory management using Reinforcement Learning, with a special emphasis on **reward function engineering**. Our goal is minimizing the waste of the products available in our supermarket.

## About Pi_Optimal

Pi_Optimal is a Python library that enables data scientists to apply Reinforcement Learning intuitively with only 7 lines of code. Key features:
- Works on tabular data
- Performs well with limited data
- Requires no prior experience in Reinforcement Learning

## Sprint Goals

1. **Develop and customize reward functions** for supermarket inventory management simulation
2. Apply Pi_Optimal to the simulation with different reward functions
3. Analyze how different reward objectives affect learned policies
4. Improve Pi_Optimal's usability by enhancing documentation and adding new models

## Setup Instructions

### 1. Clone this Repository

```bash
git clone <repository-url>
cd pyConDE-2025-sprint
```

### 2. Set Up Your Environment

We recommend using [Poetry](https://python-poetry.org/) for managing dependencies and environments, but also provide alternative options.

```bash
# Deactivate any active virtual environments (if applicable)
# For conda: conda deactivate

# Install Poetry (if you haven't already)
pipx install poetry

# Clone pi_optimal repository
git clone https://github.com/pi-optimal/pi-optimal.git
cd pi-optimal

# Install the project dependencies using Poetry
poetry install


# Activate poetry environment
poetry shell

# If it doesn't appear in interpreter lists, add the output to the path
poetry env info --path


```

## Project Structure

- `supermarket_env.py`: Contains the Gym environment for inventory management simulation
- `example_notebook.ipynb`: Jupyter notebook with examples of using Pi_Optimal with our simulation and different reward functions
- `example_building_temperature_control.ipynb`: Additional example showing Pi_Optimal applied to building temperature control
- `requirements.txt`: List of required packages

## Getting Started

1. Review the simulation environment in `supermarket_env.py`
2. Open `example_notebook.ipynb` to see how to:
   - Define and customize reward functions
   - Use Pi_Optimal with different reward objectives
   - Compare performance across reward functions
3. Choose one of the sprint tasks to work on:
   - Design a custom reward function for new business objectives
   - Experiment with different Pi_Optimal configurations
   - Contribute to Pi_Optimal documentation or features

## Reward Function Engineering

A key focus of this sprint is **reward function engineering** - the art of designing reward signals that guide reinforcement learning agents toward specific objectives.

The project includes several reward function examples:
- **Default**: Balanced profit-oriented approach
- **Stockout-Averse**: Prioritizes customer satisfaction by heavily penalizing stockouts
- **Cost-Efficient**: Focuses on minimizing operational costs
- **Freshness-Focused**: Emphasizes product freshness and reduces waste

You'll explore how modifying the reward function changes agent behavior and learn how to design rewards that align with specific business goals.

## The Inventory Management Problem

The supermarket inventory management problem involves:
- Deciding daily order quantities for products
- Balancing inventory costs against stockout risks
- Considering lead times, demand uncertainty, and perishability

The environment simulates:
- Daily customer demand (stochastic)
- Inventory holding costs
- Stockout penalties
- Ordering costs
- Product perishability

Your task is to apply Pi_Optimal with different reward functions to find optimal ordering policies for various business objectives.

## Resources

- [Pi_Optimal Documentation](https://pi-optimal.com/docs/getting-started)
- [OpenAI Gym Documentation](https://gymnasium.farama.org)