import numpy as np
from typing import List, Dict
import pandas as pd

from ItemBandit.src.bandit_item import BanditItem


class ItemMixBandit:
    def __init__(self, items: List[BanditItem], epsilon: float, epsilon_decay: float = 0.05, min_epsilon: float = 0.1, n_simulate_pulls: int = 10, ):
        """
        This class handles the selection of items based on the bandit algorithm.
        It takes a list of BanditItem objects representing the available items,
        an exploration factor (epsilon), the number of items to select (n),
        and optional constraints for item selection.

        It tracks the rewards and pulls for each item and provides a method for selecting
        the best items based on the rewards.

        Args:
            items: A list of BanditItem objects.
            epsilon: The exploration factor (probability of selecting a random item).
            n: The number of items to select.
            constraints: Constraints for item selection (e.g., maximum price, maximum number of same items).

        Returns:
            None
        """
        self.items = items
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n = len(items) 
        self.n_pulls = n_simulate_pulls
        self.rewards = np.zeros(len(items))
        self.pulls = np.zeros(len(items))
        self.ranked_performance = None
        self.selected_items = None
        self.selected_rewards = np.zeros(len(items))
        self.initial_pulls()

    def initial_pulls(self):
        """
        Perform initial random pulls to rank the items based on performance.

        Args:
            num_pulls: The number of random pulls to perform.

        Returns:
            None
        """
        for i, item in enumerate(self.items):
            reward = item.reward_function()
            self.rewards[i] += reward
            self.pulls[i] += 1

        self.ranked_performance = np.argsort(self.rewards)[::-1]  # Descending order

    def simulate_pulls(self):
        """
        Simulate the rewards from having an item available in a store. This method will 
        randomly add rewards from the item to estimate it's results

        Args:
          items: The list of selected items.
        
        Returns:
          
        """
        for item in self.selected_items:
            for _ in range(self.n_pulls):
                reward = item.reward_function()
                self.selected_rewards[self.selected_items.index(item)] += reward
                self.pulls[self.selected_items.index(item)] += 1

    def select_items(self) -> List[BanditItem]:
        """
        Select items based on the bandit algorithm.
        This selects all of the items, but returns them in the order they should be used.

        Args:
            previous_item_mix: The list of previously selected items.

        Returns:
            A list of selected BanditItem objects.
        """
        if self.ranked_performance is None:
            self.initial_pulls()
        assert self.ranked_performance is not None, 'Ranked performance is none'

        # Create a dictionary of item indices and their ranks.
        item_ranks: Dict[int, int] = {i: rank for i, rank in enumerate(self.ranked_performance)}

        selected_items: List[BanditItem] = []
        while len(selected_items) < self.n:
            # If there are no more items to select, break out of the loop.
            if not item_ranks:
                break
            
            if np.random.rand() < self.epsilon:
                # Random selection.
                item_index = np.random.choice(list(item_ranks.keys()))
            else:
                # Exploitation: select based on rewards (ranks).
                item_index = max(item_ranks.keys(), key=item_ranks.get)

            selected_items.append(self.items[item_index])

            # Remove selected item from possible choices.
            del item_ranks[item_index]

            # Apply epsilon decay so that next item is more likely to be selected via exploitation
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)

        # Once items are selected, add a reward to simulate that item being used
        self.selected_items = selected_items
        # sorted_items = self.simulate_pulls()
        self.simulate_pulls()

        return selected_items
