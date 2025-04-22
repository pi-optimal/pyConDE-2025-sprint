# pi_optimal Sprint @ PyConDE & PyData 2025

[![PyPI version](https://img.shields.io/pypi/v/pi-optimal)](https://pypi.org/project/pi-optimal)
[![License](https://img.shields.io/github/license/pi-optimal/pi-optimal)](https://github.com/pi-optimal/pi-optimal/blob/main/LICENSE)

Welcome to the **pi_optimal Hands‑On Sprint** on **Reward Function Engineering**! In this session, you'll learn how to apply reinforcement learning to supermarket inventory management—focusing on how different reward designs steer agent behavior.

---

## 🚀 Sprint Overview

- **Goal**: Minimize waste & maximize profit by experimenting with reward functions in a perishable‐goods inventory simulation.
- **Toolkit**: [pi_optimal](https://pi-optimal.com/docs/getting-started), our 7‑line RL library for tabular data.
- **Environment**: A custom Gymnasium env (`SupermarketEnv` & `MultiProductSupermarketEnv`) simulating daily demand, holding costs, stockouts, and perishability.

---

## 📦 Installation & Setup

1. **Clone this repo**  
   ```bash
      git clone https://github.com/pi-optimal/pyConDE-2025-sprint.git
      cd pyConDE-2025-sprint
   ```

2. **Install pi_optimal**  
   We recommend Poetry for dependency management:
   ```bash
      pipx install poetry
      git clone https://github.com/pi-optimal/pi-optimal.git
      cd pi-optimal
      poetry install
      poetry shell
   ```

   But if there are issues, feel free to just use the pip install command:
   ```bash
      pip install pi_optimal
   ```

---

## 📂 Project Structure
```
pyConDE-2025-sprint/
├── supermarket_env.py           # Single and multi product Gym env
├── multi_product_inventory_optimization.ipynb
└── README.md
```

---

## 🏁 Getting Started
1. **Explore the notebook**  
   Open multi_product_inventory_optimization.ipynb to see how to:
   - Instantiate the env
   - Define & swap reward functions
   - Train agents with pi_optimal
   - Compare performance metrics
2. **Sprint tasks**
   - Design a new reward function (e.g. penalize large orders, favor freshness)
   - Evaluate how policies change under your reward design
   - Contribute to docs or add a new sample model in pi_optimal

---

## 🎯 Reward Function Engineering

Reward functions are how we encode business objectives into RL. In this sprint you'll:
- Review the default profit‐based reward
- Create custom rewards (e.g., "freshness_bonus", "waste_penalty")
- Analyze agent decisions and trade‑offs under each design

---

## 🔗 Resources
- pi_optimal docs: https://pi-optimal.com/docs/getting-started
- Gymnasium API: https://gymnasium.farama.org
- Simulator: supermarket_env.py
- Example notebook: multi_product_inventory_optimization.ipynb

Happy coding, and may your rewards be well‑engineered! 🎉

