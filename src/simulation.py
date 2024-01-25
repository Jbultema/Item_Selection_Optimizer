import pandas as pd
import copy
from itertools import product
from typing import Dict, List, Union, Any
from pandera.typing import DataFrame

from ItemBandit.src.interface import run_bandits
from ItemBandit.schemas.bandit_schemas import (
    SalesHistorySchema,
    SimulationResultBaseSchema
)

def simulate_item_mix_selections_sequential(
    df: DataFrame[SalesHistorySchema], 
    location: str, 
    constraints_dict: Dict[str, Dict[Any, Any]],
    config: Dict[str, float], 
    num_simulations: int, 
    test_conditions: Dict[str, List[Any]], 
    selection_adjustment: float = 0.0
) -> DataFrame[SimulationResultBaseSchema]:
    """
    Performs sequential simulated item-mix choices for a single location, testing a range of condition sequentially (i.e. it does not test the combinations)
    The item-mix is passed from the starting simulation to subsequent simulations.
    A selection_adjustment can be applied to applies in a selected item-mix to simulate improved or worse performance.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing item information for all locations.
    location (str): The location to run simulations for.
    constraints_dict (dict): Dictionary with constraints for the simulations, can include keys like 'n_items', 'max_price', etc.
    config (dict): Configuration parameters for the bandit algorithm, such as 'epsilon', 'epsilon_decay', 'min_epsilon'.
    num_simulations (int): The number of simulations to run for each condition.
    test_conditions (dict): Dictionary where keys are the parameters to vary in testing, and values are lists of values to test for each parameter.
    selection_adjustment (float, optional): Adjustment to apply to the 'SELL_THROUGH_RATE' of selected items after each simulation. Positive values increase the rate, and negative values decrease it. Default is 0.0.

    Returns:
    pd.DataFrame: A DataFrame with the results of all simulations.
    """

    # Deep copy the constraints_dict and config to avoid modifying the original
    sim_constraints = copy.deepcopy(constraints_dict)
    sim_config = copy.deepcopy(config)

    selected_mixes = []
    mix_duplicates = []

    # Get data for specific location
    base_df = df[df['Location'] == location].copy()

    # Iterate over the test_conditions dictionary
    for condition, values in test_conditions.items():
        # Reset tmp_df for the new test condition
        tmp_df = base_df.copy()
        
        for value in values:
            # Check and update the relevant key in sim_constraints or sim_config
            if condition in sim_constraints:
                sim_constraints[condition] = value
            elif condition in sim_config:
                sim_config[condition] = value
            else:
                raise KeyError(f"'{condition}' not found in 'sim_constraints' or 'sim_config'")

            # Initialize previous_sim_mix for the current test condition
            previous_sim_mix: List[str] = []

            # Run the simulations for the current test condition
            for i in range(1, num_simulations + 1):
                new_mix = run_bandits(df=tmp_df, location_constraints=sim_constraints, previous_item_mix=previous_sim_mix, config=sim_config)
                
                # Adjust SELL_THROUGH_RATE in tmp_df based on selected items and selection_adjustment
                for item in new_mix['Selected Item'].unique():
                    tmp_df.loc[tmp_df['Item Name'] == item, 'SELL_THROUGH_RATE'] *= (1 + selection_adjustment)
                
                duplicates = list(set(new_mix['Selected Item'].unique().tolist()).intersection(set(previous_sim_mix)))
                print(f"Running simulation: {location} - {i} - {condition}: {value} - Fraction repeated: {len(duplicates) / new_mix['Selected Item'].nunique()}")

                new_mix[condition] = value  # add the current test condition value to the results DataFrame
                new_mix['simulation'] = i
                
                selected_mixes.append(new_mix)
                mix_duplicates.append(duplicates)
                
                previous_sim_mix = new_mix['Selected Item'].unique().tolist()

    # Combine all simulation results into a single DataFrame
    selected_df = pd.concat(selected_mixes)
    
    return selected_df, mix_duplicates


def simulate_item_mix_selections_combinatorial(
    df: DataFrame[SalesHistorySchema], 
    location: str, 
    constraints_dict: Dict[str, Dict[Any, Any]], 
    config: Dict[str, float], 
    num_simulations: int, 
    test_conditions: Dict[str, List[Any]], 
    selection_adjustment: float = 0.0
) -> DataFrame[SimulationResultBaseSchema]:
    """
    Performs sequential simulated item-mix choices for a single location, testing a range of conditions combinatorially
    The item-mix is passed from the starting simulation to subsequent simulations.
    A selection_adjustment can be applied to applies in a selected item-mix to simulate improved or worse performance.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing item information for all locations.
    location (str): The location to run simulations for.
    constraints_dict (dict): Dictionary with constraints for the simulations, can include keys like 'n_items', 'max_price', etc.
    config (dict): Configuration parameters for the bandit algorithm, such as 'epsilon', 'epsilon_decay', 'min_epsilon'.
    num_simulations (int): The number of simulations to run for each condition.
    test_conditions (dict): Dictionary where keys are the parameters to vary in testing, and values are lists of values to test for each parameter.
    selection_adjustment (float, optional): Adjustment to apply to the 'SELL_THROUGH_RATE' of selected items after each simulation. Positive values increase the rate, and negative values decrease it. Default is 0.0.

    Returns:
    pd.DataFrame: A DataFrame with the results of all simulations.
    """
    # Deep copy the constraints_dict and config to avoid modifying the original
    sim_constraints = copy.deepcopy(constraints_dict)
    sim_config = copy.deepcopy(config)

    selected_mixes = []
    mix_duplicates = []

    # Get data for specific location
    base_df = df[df['Location'] == location].copy()
    
    # Create list of all combinations of test conditions
    all_combinations = list(product(*test_conditions.values()))

    # Iterate over all combinations
    for combination in all_combinations:
        # Reset tmp_df for the new test combination
        tmp_df = base_df.copy()
        
        # Update sim_config and sim_constraints with the current combination of conditions
        for i, condition in enumerate(test_conditions.keys()):
            if condition in sim_constraints:
                sim_constraints[condition] = combination[i]
            elif condition in sim_config:
                sim_config[condition] = combination[i]
            else:
                raise KeyError(f"'{condition}' not found in 'sim_constraints' or 'sim_config'")

        # Initialize previous_sim_mix for the current test condition
        previous_sim_mix: List[str] = []

        # Run the simulations for the current test condition
        for i in range(1, num_simulations + 1):
            new_mix = run_bandits(df=tmp_df, location_constraints=sim_constraints, previous_item_mix=previous_sim_mix, config=sim_config)
            
            # Adjust SELL_THROUGH_RATE in tmp_df based on selected items and selection_adjustment
            for item in new_mix['Selected Item'].unique():
                tmp_df.loc[tmp_df['Item Name'] == item, 'SELL_THROUGH_RATE'] *= (1 + selection_adjustment)
            
            duplicates = list(set(new_mix['Selected Item'].unique().tolist()).intersection(set(previous_sim_mix)))
            print(f"Running simulation: {location} - {i} - {combination} - Fraction repeated: {len(duplicates) / new_mix['Selected Item'].nunique()}")

            for j, condition in enumerate(test_conditions.keys()):  # add the current test condition values to the results DataFrame
                new_mix[condition] = combination[j] 
            new_mix['simulation'] = i

            selected_mixes.append(new_mix)
            mix_duplicates.append(duplicates)

            previous_sim_mix = new_mix['Selected Item'].unique().tolist()
            
    # Combine all simulation results into a single DataFrame
    selected_df = pd.concat(selected_mixes)
      
    return selected_df, mix_duplicates
