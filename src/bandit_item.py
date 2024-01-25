import numpy as np
from typing import List, Dict
import scipy.stats as stats
import signal
import pandas as pd
import warnings
# # Suppress RuntimeWarnings when fitting distributions
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ItemBandit.constants.bandit_constants import (
    DEFAULT_TIMEOUT_DURATION,
    DEFAULT_BOOTSTRP_OPTION,
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_ITEM_PRICE,
    DEFAULT_SELL_THROUGH_RATE
)


class BanditItem:
    def __init__(
        self,
        name: str,
        item_id: str,
        price: float = DEFAULT_ITEM_PRICE,
        sell_through_rate_history: List[float] = DEFAULT_SELL_THROUGH_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        timeout_duration: int = DEFAULT_TIMEOUT_DURATION,
        bootstrap: bool = DEFAULT_BOOTSTRP_OPTION,
    ):
        """
        This class stores information about an item, including its name, price, sell-through rate history,
        and the discount factor used for reward calculation. It provides methods for selecting the best
        distribution for the item's sell-through rate, calculating the reward for the item, and handling
        timeout when fitting distributions.

        Args:
            name: The name of the item.
            price: The price of the item.
            sell_through_rate_history: A list of historical sell-through rates for the item.
            discount_factor: The discount factor used in reward calculation.
            timeout_duration: The duration in seconds before timeout when fitting distributions.

        Returns:
            None
        """
        self.name = name
        self.item_id = item_id
        self.price = price
        self.sell_through_rate_history = sell_through_rate_history
        self.discount_factor = discount_factor
        self.timeout_duration = timeout_duration
        self.distribution = None
        self.params = None
        self.bootstrap = bootstrap

    def get_name(self) -> str:
        """
        Get the name of the item.

        Returns:
            The name of the item.
        """
        return self.name

    def get_id(self) -> str:
        """
        Get the name of the item.

        Returns:
            The name of the item.
        """
        return self.item_id

    def get_price(self) -> float:
        """
        Get the price of the item.

        Returns:
            The price of the item.
        """
        return self.price

    def get_distribution(self):
        """
        Get the distribution associated with the item.

        Returns:
            The distribution associated with the item.
        """
        return self.distribution

    def get_params(self):
        """
        Get the parameters of the distribution associated with the item.

        Returns:
            The parameters of the distribution associated with the item.
        """
        return self.params

    def select_best_distribution(self):
        """
        Select the best distribution for the item based on historical sell-through rates.
        Includes logic for creating a fake distribution for items with a single record.

        Returns:
            None
        """
        sell_through_rates = np.ravel(self.sell_through_rate_history)
        if len(sell_through_rates) <= 1:
            noise = np.random.normal(loc=0.1, scale=0.1, size=10)
            fake_distribution = (sell_through_rates[0] + noise) * self.discount_factor
            self.distribution = stats.norm
            self.params = (np.mean(fake_distribution), np.std(fake_distribution))
        else:
            distributions = [
                stats.norm, stats.expon, stats.gamma,
                stats.beta, stats.weibull_min, stats.weibull_max,
                stats.lognorm, stats.powerlaw, stats.poisson
            ]
            best_distribution = distributions[0]
            best_params = distributions[0].fit(sell_through_rates)
            best_ks_stat = np.inf

            for distribution in distributions:
                try:
                    signal.signal(signal.SIGALRM, self.timeout_handler)
                    signal.alarm(self.timeout_duration)
                    try:
                        params = distribution.fit(sell_through_rates)
                    except:
                        continue
                    finally:
                        signal.alarm(0)

                    ks_stat = stats.kstest(sell_through_rates, distribution.name, args=params)[0]

                    if ks_stat < best_ks_stat:
                        best_distribution = distribution
                        best_params = params
                        best_ks_stat = ks_stat
                except TimeoutError:
                    continue
                finally:
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)

            self.distribution = best_distribution
            self.params = best_params

    def timeout_handler(self, signum, frame):
        """
        Timeout handler for distribution fitting.

        Args:
            signum: The signal number.
            frame: The current stack frame.

        Raises:
            TimeoutError: When the distribution fitting times out.

        Returns:
            None
        """
        raise TimeoutError("Distribution fitting timed out")

    def reward_function(self) -> float:
        """
        Calculate the reward for the item based on the selected distribution.
        This approach bootstraps from the distribution and returns the average.

        Returns:
            The reward for the item.
        """

        if self.distribution is None:
            self.select_best_distribution()
        assert self.distribution is not None, f'Distribution Error for BanditItem {self.name}'

        if self.bootstrap:
            bootstrap_sample = self.distribution.rvs(*self.params, size=25)
            reward = np.mean(bootstrap_sample)
            reward = (reward - min(bootstrap_sample)) / (max(bootstrap_sample) - min(bootstrap_sample))

        else:
            reward = self.distribution.rvs(*self.params)
            while reward < 0 or reward > 1:
                reward = self.distribution.rvs(*self.params)

        return reward
