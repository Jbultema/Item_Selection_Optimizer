import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, Union, Dict, List, Any, Optional
from pandera.typing import DataFrame

from ItemBandit.data.queries import (
    query_snowflake,
    get_item_mix_bandit_data_query,
    get_previous_items_delivered
)
from ItemBandit.schemas.bandit_schemas import (
    SalesHistorySchema,
    AggregatedSalesHistorySchema,
    LocationConstraint,
    PreviousItemHistorySchema
)
from ItemBandit.constants.bandit_constants import (
    DEFAULT_MAX_PRICE,
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_NUM_OBSERVATIONS,
    DEFAULT_NUM_PREVIOUS,
    DEFAULT_MAX_SELL_THROUGH
)


def get_item_mix_bandit_data(
    end_date: str = dt.date.today().strftime('%Y-%m-%d'),
    history_length: int = DEFAULT_HISTORY_LENGTH,
    num_observations: int = DEFAULT_NUM_OBSERVATIONS
) -> pd.DataFrame:
    """
    Retrieve item mix bandit data from the database tables.

    Args:
        end_date: End date for the query (default: today's date).
        history_length: Length of history to consider (default: 56).
        num_observations: Number of most recent observations to include (default: 20).

    Returns:
        A DataFrame containing the item mix bandit data.
    """

    query = get_item_mix_bandit_data_query(end_date, history_length)
    df = query_snowflake(query)
    # logic to cap max sell_through_rate
    df['SELL_THROUGH_RATE'] = np.where(
        (df['TOTAL_SOLD'] >= df['TOTAL_DELIVERED']) | (df['SELL_THROUGH_RATE'] > DEFAULT_MAX_SELL_THROUGH), 
        1.0, 
        df['SELL_THROUGH_RATE']
    )
    # restrict to the most recent num_observations
    df = df.groupby('Series').tail(num_observations)

    return df


def preprocess_data(
    df: DataFrame[SalesHistorySchema]
) -> Tuple[DataFrame[AggregatedSalesHistorySchema], DataFrame[SalesHistorySchema]]:
    """
    Preprocess the data by filling missing sell-through rates with category or location averages.

    Args:
        df: The DataFrame containing the data.

    Returns:
        The preprocessed DataFrame.
    """
    raw_df = df.copy()
    item_avg_sell_through = df.groupby('Item Name')['SELL_THROUGH_RATE'].mean()
    category_avg_sell_through = df.groupby('Category')['SELL_THROUGH_RATE'].mean()
    location_avg_sell_through = df.groupby('Location')['SELL_THROUGH_RATE'].mean()
    global_sell_through = df['SELL_THROUGH_RATE'].mean()
    
    df = df.groupby(['Location', 'Location Id', 'Category', 'Item Category', 'Item Name', 'Item Id'], as_index=False).agg({'Price': 'max', 'SELL_THROUGH_RATE': list, 'TOTAL_DELIVERED': 'sum', 'TOTAL_SOLD': 'sum'})
    df['SELL_THROUGH_FILL'] = df['SELL_THROUGH_RATE'].apply(lambda x: not x or np.isnan(x).any() or np.sum(x) == 0)
    df['FILL_SOURCE'] = ''
    
    df[['SELL_THROUGH_RATE', 'FILL_SOURCE']] = df.apply(
        lambda row: fill_sell_through_rate(row, item_avg_sell_through, category_avg_sell_through, location_avg_sell_through, global_sell_through),
        axis=1, result_type='expand'
    )
    
    return (df, raw_df)


def fill_sell_through_rate(
    row: pd.Series, item_avg_sell_through: Dict[Any, Union[float, None]], 
    category_avg_sell_through: Dict[Any, Union[float, None]], 
    location_avg_sell_through: Dict[Any, Union[float, None]], 
    global_sell_through: float
) -> pd.Series:
    """
    Fill missing sell-through rates with Item, category, location, or global averages.

    Args:
        row: The row of the DataFrame.
        item_avg_sell_through: The dictionary containing average sell-through rates for each item.
        category_avg_sell_through: The dictionary containing average sell-through rates for each category.
        location_avg_sell_through: The dictionary containing average sell-through rates for each location.
        global_sell_through: The average sell-through rate across all data.

    Returns:
        The filled sell-through rate and fill source.
    """
    sell_through_rate = row['SELL_THROUGH_RATE']
    fill_source = None

    if not sell_through_rate or np.isnan(sell_through_rate).all() or np.sum(sell_through_rate) == 0:
        item_name = row['Item Name']
        category = row['Category']
        location = row['Location']
        
        item_rate = item_avg_sell_through.get(item_name, None)
        category_rate = category_avg_sell_through.get(category, None)
        location_rate = location_avg_sell_through.get(location, None)
        
        if item_rate is not None:
            fill_value = item_rate
            fill_source = 'Item'
        elif category_rate is not None:
            fill_value = category_rate
            fill_source = 'Category'
        elif location_rate is not None:
            fill_value = location_rate
            fill_source = 'Location'
        else:
            fill_value = global_sell_through
            fill_source = 'Global'
        
        sell_through_rate = fill_value
    
    return pd.Series([sell_through_rate, fill_source])


def process_previous_orders(
    current_data: DataFrame[SalesHistorySchema],
    previous_order_df: Optional[DataFrame[PreviousItemHistorySchema]] = None,
    previous_order_path: Optional[str] = None,
    default_repeat_items: int = DEFAULT_NUM_PREVIOUS,
    default_max_price: float = DEFAULT_MAX_PRICE,
    default_increase_orders: bool = True
) -> Tuple[Dict, Dict]:
    """
    Process previous orders and generate the previous orders dictionary and location constraints dictionary.

    Args:
        current_data: DataFrame containing the current data.
        previous_order_path: Path to the previous order file.
        default_repeat_items: Default number of previous items to consider (default: 0).
        default_max_price: Default maximum price constraint (default: 19.99).
        default_increase_orders: Default flag for increasing repeat orders (default: False).

    Returns:
        A tuple containing the previous orders dictionary and location constraints dictionary.
    """
    if previous_order_df is not None:
        previous_orders = previous_order_df
    elif previous_order_path is not None:
        previous_orders = pd.read_csv(previous_order_path)[['Location Name', 'Item Name']].drop_duplicates().rename(columns={'Location Name': 'Location'})
    else:
        previous_orders = get_previous_items_delivered()
        
    # Create the previous orders dictionary
    previous_orders_dict = previous_orders.groupby('Location').agg(list).to_dict()['Item Name']

    # Add category info and determine the number of items per category per location
    location_category_count = previous_orders.copy()
    location_category_count = location_category_count.merge(current_data[['Location', 'Category', 'Item Name']].drop_duplicates(), how='left', on=['Location', 'Item Name'])
    location_category_count = location_category_count.groupby(['Location', 'Category']).count().reset_index()

    # Create a dictionary to store the transformed structure
    location_constraints_dict: Dict[str, LocationConstraint] = {}

    for _, row in location_category_count.iterrows():
        location = row['Location']
        category = row['Category']

        # Check if the location exists in the result dictionary
        if location not in location_constraints_dict:
            location_constraints_dict[location] = {'n_items': {}, 'max_price': np.nan, 'increase_repeats': False, 'n_previous_items': 0}

        # Assign the actual count of items in a specific category-location combination to the corresponding dictionary key
        location_constraints_dict[location]['n_items'][category] = row['Item Name']

    # Set max_price based on location name
    for location in location_constraints_dict.keys():
        if 'costco' in location.lower():
            location_constraints_dict[location]['max_price'] = 4.99
        else:
            location_constraints_dict[location]['max_price'] = default_max_price

    # Add default values for 'n_previous_items' and 'increase_repeats'
    for location in location_constraints_dict.keys():
        location_constraints_dict[location]['n_previous_items'] = default_repeat_items
        location_constraints_dict[location]['increase_repeats'] = default_increase_orders

    return previous_orders_dict, location_constraints_dict
