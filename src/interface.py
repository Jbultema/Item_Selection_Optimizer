import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from pandera.typing import DataFrame

from ItemBandit.src.bandit_item import BanditItem
from ItemBandit.src.bandit_manager import CategoryBanditManager
from ItemBandit.src.data_prep import preprocess_data
from ItemBandit.schemas.bandit_schemas import (
    SalesHistorySchema,
    SelectedItemsSchema,
    SelectedItemsLocationsSchema
)


def run_bandits(
    df: DataFrame[SalesHistorySchema],
    location_constraints: Dict[str, Dict[Any, Any]],
    previous_item_mix: List[str],
    config: Dict[str, float],
    save_results: bool = False,
    save_path: str = None,
) -> DataFrame[SelectedItemsSchema]:
    """
    This function takes a DataFrame containing the data, constraints for item selection at each location,
    previous item mixes for each location, and configuration parameters for the bandit algorithm.

    It preprocesses the data, creates BanditItem objects for each item, initializes a CategoryBanditManager,
    and runs the bandit algorithm to select items. It returns a DataFrame containing the selected items and
    their rewards at each location and category.

    Args:
        df: The DataFrame containing the data.
        location_constraints: Constraints for item selection at each location.
        previous_item_mix: The dictionary of previous item mixes for each location.
        config: Configuration parameters for the bandit algorithm.
        save_results: Flag indicating whether to save the results to a file.
        save_path: The path to save the results (optional).

    Returns:
        The DataFrame containing the selected items, prices, and rewards.
    """
    df, raw_df = preprocess_data(df)
    categories_items: Dict[str, List[BanditItem]] = {}
    previous_items: Dict[str, List[BanditItem]] = {}
    
    for _, row in df.iterrows():
        category = row['Category']
        item_name = row['Item Name']
        item_id = row['Item Id']
        price = row['Price']
        sell_through_rate = row['SELL_THROUGH_RATE']

        if category not in location_constraints['n_items']:
            continue
        if category not in categories_items:
            categories_items[category] = []
        if category not in previous_items:
            previous_items[category] = []

        item = BanditItem(name=item_name, item_id=item_id, price=price, sell_through_rate_history=[sell_through_rate])
        categories_items[category].append(item)
        
        if item.name in previous_item_mix:
            previous_items[category].append(item)

    manager = CategoryBanditManager(
        items=categories_items,
        config=config,
        previous_items=previous_items,
        location_constraints=location_constraints,
        salesdata=raw_df, 
    )

    # Ensure all categories are initialized in the CategoryBanditManager
    for category in location_constraints['n_items']:
        manager.get_items_for_category(category)

    result_data = []  # List to store results for each category
    for category, _ in categories_items.items():
        selected = manager.select_items(category)
        for item in selected:
            item_reward = item.reward_function()            
            result_data.append({
                'Category': category,
                'Selected Item': item.get_name(),
                'Item Id': item.get_id(),
                'Price': item.get_price(), 
                'Item Reward': item_reward,
            })

    result_df = pd.DataFrame(result_data)
    result_df['Category Reward'] = result_df.groupby('Category')['Item Reward'].transform('mean')

    if save_results:
        result_df.to_csv(save_path, index=False)

    return result_df
  

def run_bandits_multiple_locations(
    df: DataFrame[SalesHistorySchema],
    location_constraints: Dict[str, Dict[Any, Any]],
    previous_item_dict: Dict[str, List[str]],
    config: Dict[str, float],
    save_results: bool = False,
    save_path: str = None
) -> DataFrame[SelectedItemsLocationsSchema]:
    """
    This function runs the bandit algorithm for multiple locations and appends the results to a DataFrame.

    Args:
        df: The DataFrame containing the data.
        location_constraints: Constraints for item selection at each location.
        previous_item_mix: The dictionary of previous item mixes for each location.
        config: Configuration parameters for the bandit algorithm.
        save_results: Flag indicating whether to save the results to a file.
        save_path: The path to save the results (optional).

    Returns:
        The DataFrame containing the selected items and rewards for each location.
    """
    result_data = []  # List to store results for each location

    # restrict data to location in constraints
    start_size = df['Location'].unique()
    df = df[df['Location'].isin(location_constraints.keys())].copy()
    end_size = df['Location'].unique()
    if end_size.shape[0] < start_size.shape[0]:
        print(f'There were {start_size.shape[0]} locations in the data but {start_size.shape[0] - end_size.shape[0]} locations not present in constraints: {list(set(start_size).difference(set(end_size)))}')

    for location, constraints in tqdm(location_constraints.items()):
        tmp = df[df['Location'] == location]
        selected_mix = run_bandits(tmp, constraints, previous_item_dict.get(location, []), config)
        if selected_mix.empty:
            print(f'No Items selected at {location}!')
        else:
            selected_mix['Location'] = location
            selected_mix['Location Id'] = tmp['Location Id'].unique()[0]
            result_data.append(selected_mix)
    
    # organize results
    result_df = pd.concat(result_data)
    result_df['Location Reward'] = result_df.groupby('Location')['Item Reward'].transform('mean')
    result_df = result_df[['Location', 'Location Id', 'Category', 'Selected Item', 'Item Id', 'Price', 'Item Reward', 'Category Reward', 'Location Reward']]
    if save_results:
        result_df.to_csv(save_path, index=False)

    return result_df
