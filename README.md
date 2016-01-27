# reinforcement_learning
Playing around with Reinforcement Learning, running simulations.

Based on "Reinforcement Learning - An Introduction" by Sutton and Brato.

Currently has a Monte Carlo black jack simulation, and an N Bandits simulation using different Agent algorithms.

# Setup
```bash
conda env create
source activate reinf
```

# Running
```bash
python src/monte_carlo_blackjack.py

python src/runner.py simulations/<your_simulation>.yml
```

# Configuration
All settings are in the settings/*.yml files, and should be self-evident
