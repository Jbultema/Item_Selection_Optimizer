# Multi-Armed Bandit Module for Item Mix Optimization

## Overview
This Python module implements a sophisticated Multi-Armed Bandit (MAB) system for item mix optimization. 
It's designed to dynamically allocate resources or select options (items) with the highest potential for 
reward based on probabilistic modeling and user-defined constraints. The module is particularly useful in 
scenarios where there are multiple options to choose from, and the goal is to maximize a certain objective 
(like click-through rates, conversions, etc.) under specific constraints.

## Key Features
- **Multi-Armed Bandit Algorithm**: Utilizes MAB algorithms for optimizing the selection of items based on their performance.
- **Thompson Sampling**: Employs Thompson Sampling, a probability-based technique, for deciding which arm (item) to pull next. 
  This method balances exploration (trying new or less understood items) and exploitation (leveraging known high-performing items).
- **Custom User Constraints**: Allows for the integration of user-defined constraints, enabling the algorithm to operate within 
  specified parameters or rules set by the user.

## Module Architecture
The architecture comprises several components working together to implement the MAB approach:

- `bandit_item.py`: Defines `BanditItem`, the basic unit in the bandit algorithm. Each item represents an 'arm' in the MAB context 
  with its own probability distribution based on its performance.

- `item_mix_bandit.py`: Contains the `CategoryBandit` class, which manages a group of `BanditItem` instances. It applies Thompson Sampling 
  to these items, continually adjusting to their performance as more data becomes available.

- `bandit_manager.py`: A management layer for handling multiple `CategoryBandit` instances, facilitating the optimization process 
  across diverse item categories.

- `interface.py`: Features the `run_bandits()` function as the primary entry point for executing the bandit algorithms. It integrates 
  the entire process, respecting the user-defined constraints and making decisions based on the Thompson Sampling strategy.

## Installation
```bash
git clone https://github.com/Jbultema/ItemBandit.git
cd ItemBandit
pip install -r requirements.txt
```

## Author
Jarred Bultema, PhD