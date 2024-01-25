import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import Levenshtein
import spacy

from bandit_item import BanditItem
from item_mix_bandit import ItemMixBandit
from ItemBandit.constants.bandit_constants import (
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    DEFAULT_NUM_ITEMS,
    DEFAULT_MAX_PRICE,
    DEFAULT_NUM_PREVIOUS,
    DEFAULT_INCREASE_REPEATS,
    DEFAULT_NUM_CATEGORY_ITEMS,
    DEFAULT_MAX_LEVENSHTEIN_DISTANCE_SAME_ITEM,
    DEFAULT_MIN_COSINESIMILARITY_SAME_ITEM,
    DEFAULT_PROTECT_CORE_PRODUCTS,
    DEFAULT_MAX_CORE_PRODUCTS,
    DEFAULT_DETECT_CORE_PRODUCT_SENSITIVITY,
    DEFAULT_NUM_ITEM_BUFFER_FROM_CORE,
    DEFAULT_LOOKBACK_WINDOW
)

class CategoryBanditManager:
    def __init__(self, items: Dict, config: Dict, previous_items: Dict, location_constraints: Dict, salesdata: pd.DataFrame):
        """
        This class manages multiple CategoryBandit instances, each representing a different category.
        It takes an exploration factor (epsilon), a decay rate for the exploration factor,
        and optional constraints for item selection.

        Args:
            items: dictionary with Category-name keys, and values that are BanditItem instances of all eligibile items in that category
            config: dictionary with global-settings to be used for all categories within a location and across locations
            previous_items: dictionary of BanditItem instances that were on the previous/current item-mix
            location_constraints: dictionary with location/location-category level configuations to control the item-mix process
            salesdata: dataframe with the delivery-period aggregate sales and delivery data
        """
        self.items = items
        self.previous_items = previous_items
        self.chosen_items: List = []
        self.core_products: List = []
        self.bandits: dict = {}
        self.protect_core_products = config.get('core_products', DEFAULT_PROTECT_CORE_PRODUCTS)
        self.core_product_sensitivity = config.get('core_product_sensitivity', DEFAULT_DETECT_CORE_PRODUCT_SENSITIVITY)
        self.lookback_window = config.get('lookback_window', DEFAULT_LOOKBACK_WINDOW)
        self.max_core_products = config.get('max_core_products', DEFAULT_MAX_CORE_PRODUCTS)
        self.num_item_buffer_from_core = config.get('num_item_buffer_from_core', DEFAULT_NUM_ITEM_BUFFER_FROM_CORE)
        self.epsilon = config.get('epsilon', DEFAULT_EPSILON)
        self.epsilon_decay = config.get('epsilon_decay', DEFAULT_EPSILON_DECAY)
        self.min_epsilon = config.get('min_epsilon', DEFAULT_MIN_EPSILON)
        self.num_items = location_constraints.get('n_items', DEFAULT_NUM_ITEMS)
        self.max_price = float(location_constraints.get('max_price', DEFAULT_MAX_PRICE))
        self.num_previous_items = location_constraints.get('n_previous_items', DEFAULT_NUM_PREVIOUS)
        self.increase_repeats = location_constraints.get('increase_repeats', DEFAULT_INCREASE_REPEATS)

        self.nlp = spacy.load("en_core_web_md")
        self.verbose = False  # for debugging
        # process the sales data
        self.salesdata = self._process_sales_data(salesdata)

    def _process_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates the dataframe in self.salesdata to the recent lookback window.
        This data is used for core-product detection
        """
        grouper_key = ['Location', 'Location Id', 'Category', 'Item Id', 'Item Name', 'Series']
        df['PERIOD_START_DATE'] = pd.to_datetime(df['PERIOD_START_DATE'])
        # restrict to recent data
        # group by series and aggregate
        df_agg = df[df['PERIOD_START_DATE'] >= (df['PERIOD_START_DATE']).max() - pd.DateOffset(days=self.lookback_window)].groupby(
            grouper_key, 
            as_index=False
        ).agg(
            total_periods=('PERIOD_START_DATE', 'nunique'),
            number_deliveres=('TOTAL_DELIVERED', 'count'),
            total_delivered=('TOTAL_DELIVERED', 'sum'),
            num_sales=('TOTAL_SOLD', 'count'),
            total_sales=('TOTAL_SOLD', 'sum'),
            avg_sell_through_rate=('SELL_THROUGH_RATE', 'mean')
        )
        df_agg['average_sales'] = df_agg['total_sales'] / df_agg['num_sales']
        df_agg.sort_values(['Location', 'total_sales', 'average_sales'], ascending=[True, False, False], inplace=True)

        # Add rank for TOTAL_SALES per Location
        df_agg['sales_rank'] = df_agg.groupby('Location')['total_sales'].rank(ascending=False, method='first').astype(int)

        return df_agg

    def get_items_for_category(self, category: str) -> Optional[ItemMixBandit]:
        """
        Get the ItemMixBandit instance for a given category.

        Args:
            category: The category name.

        Returns:
            The ItemMixBandit instance for the category, or None if the category is not present in the constraints.
        """
        if category in self.num_items:
            self.bandits[category] = ItemMixBandit(
                items=self.items[category], 
                epsilon=self.epsilon, 
                epsilon_decay=self.epsilon_decay, 
                min_epsilon=self.min_epsilon
            )
        return self.bandits.get(category)
      
    def select_items(self, category: str) -> List[BanditItem]:
        """
        Select a list of items from a given category, applying constraints and reducing to n items.

        Args:
            category: The category name.

        Returns:
            A list of selected BanditItem objects.
        """
        category_bandit = self.bandits[category]
        num_items = min(self.num_items.get(category, DEFAULT_NUM_CATEGORY_ITEMS), len(self.bandits[category].items))
        selected_items = category_bandit.select_items() 

        # protect core-products if they exist
        # check if core-products are in this category
        # ensures that we aren't sending too many core-products
        # then clean up the types and order of items
        core_items_names = self.detect_core_products()
        potential_core_items = [x for x in selected_items if x.name in core_items_names]
        if len(potential_core_items) > num_items - self.num_item_buffer_from_core:
            potential_core_items = potential_core_items[:num_items - self.num_item_buffer_from_core]
        self.core_items = potential_core_items 
        filtered_selected_items = [x for x in selected_items if x not in self.core_items]
        queued_items = self.core_items + filtered_selected_items

        # apply business constraints to item mix
        # lower-idx items are given priority over higher-idx items for some steps
        constrained_items = self.apply_constraints(queued_items, category)
        reduced_items = constrained_items[:num_items]

        return reduced_items

    def detect_core_products(self) -> List:
        """
        Triggers data aggregations and then checks if any core-products are detected.
        """
        # check for core products
        if self.protect_core_products:
            core_items = self._identify_core_products_via_sensitivity()
        else:
            core_items = []

        return core_items

    def _identify_core_products_via_sensitivity(self) -> List:
        """
        This function analyzes the item-level cumulative sales data to determine the relative
        contribution to total sales of each item, in a total-sales ranked order. After determining
        the relative contribution, this function determines if any of the top-ranked items are 
        "core products" with outsized percentage of sales.

        The single control for threshold of the "core product" determination is the sensitivity, which
        controls the number of items able to be "core products" and the performance threshold required 
        for those items relative to the non "core products".

        The data provided needs to be filtered for the appropriate comparison, as no data filtering 
        happens within the function.

        Arguments:
        df: dataframe containing the aggregated sales data at the item-location row-level granularity with
            'TOTAL_SALES' and 'Item Name' columns.
        sensitivity: how easy it is for an item to be a "core product". Low sensitivity makes it easier,
            high sensitivity makes it more difficult to be a "core product"

        Returns:
        cdf_df: dataframe with CDF results where each row represents an 'Item Name', 'rank', and relative sales data
        """
        df = self.salesdata.copy() 
        df_sorted = df.sort_values('total_sales', ascending=False).reset_index(drop=True)
        total_sales = df['total_sales'].sum()
        # calculate CDF of sales by each items in sales-rank order
        cumulative_sales = [df_sorted['total_sales'].iloc[:i + 1].sum() for i in range(len(df))]
        cumulative_proportion = [sales_sum / total_sales for sales_sum in cumulative_sales]
        slopes = [0] + [(cumulative_proportion[i] - cumulative_proportion[i - 1]) for i in range(1, len(df))]
        
        cdf_df = pd.DataFrame({
            'Item Name': df_sorted['Item Name'],
            'rank': range(1, len(df) + 1),
            'cumulative_sales': cumulative_sales,
            'cumulative_proportion': cumulative_proportion,
            'slope': slopes
        })
        cdf_df['is_core'] = False
        
        # Dynamically determining the core_rank_threshold and slope_difference_threshold_ratio based on sensitivity
        core_rank_threshold = max(1, int(self.max_core_products * (1 - self.core_product_sensitivity)))
        slope_difference_threshold_ratio = 1 + (self.core_product_sensitivity * 2)
        max_slope_core_candidate = cdf_df.loc[cdf_df['rank'] <= core_rank_threshold, 'slope'].max()
        max_slope_noncore_candidate = cdf_df.loc[cdf_df['rank'] > core_rank_threshold, 'slope'].max()

        if max_slope_core_candidate / max_slope_noncore_candidate >= slope_difference_threshold_ratio:
            core_item_rank = cdf_df.loc[cdf_df['slope'] == max_slope_core_candidate, 'rank'].values[0]
            cdf_df.loc[cdf_df['rank'] <= core_item_rank, 'is_core'] = True

        potential_core_items = cdf_df[cdf_df['is_core']]['Item Name'].unique().tolist()
        
        return potential_core_items 

    def _apply_semantic_constraints(self, items, semantic_threshold: int = DEFAULT_MAX_LEVENSHTEIN_DISTANCE_SAME_ITEM):
        """
        Check to make sure that no items with extremely similar names are selected.
        This check uses a letter-replacement approach (measured by Levenshtein distance) to find nearly 
        identically named items, ex: Chicken Salad on White Bread vs Chicken Salad on Wheat Bread.

        Upon duplicate identification, the first-occurrence of the name is kept, and subsequent
        occurrences are removed from the list of items.

        Args:
            items: list of bandit items
            semantic_thresold: the number of character-differences between the two names

        Returns:
            A list of bandit items, with detected duplicates removed.
        """
        unique_items: list = []
        for item1 in items:
            is_unique = True
            for item2 in unique_items:
                distance = Levenshtein.distance(item1.name, item2.name)
                if distance <= semantic_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_items.append(item1)

        return unique_items

    def _vectorize(self, text):
        """
        Converts a string to a vector using spacy model (currently medium English-language model)
        """
        return self.nlp(text).vector

    def _cosine_similarity(self, vec1, vec2):
        """
        Calculates cosine similarity between two vectors, representing string embeddings
        """
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2) 
        return np.dot(vec1, vec2) / denom if denom > 0 else 0

    def _apply_semantic_constraints_using_embeddings(self, items, semantic_threshold: float = DEFAULT_MIN_COSINESIMILARITY_SAME_ITEM):
        """
        Check to make sure that no items with extremely similar names are selected, using
        cosine_similarity for scoring compared to a threshold.
        This check is designed to capture the more subtle name interactions.

        Upon duplicate identification, the first-occurrence of the name is kept, and subsequent
        occurrences are removed from the list of items.

        Args:
            items: list of bandit items
            semantic_threshold: the cosine_similarity threshold (scale -1 <-> 1, 1 == identical, -1 == opposite)

        Returns:
            A list of bandit items, with detected duplicates removed.
        """
        unique_items: list = []
        for item1 in items:
            is_unique = True
            vec1 = self._vectorize(item1.name)
            for item2 in unique_items:
                vec2 = self._vectorize(item2.name)
                similarity = self._cosine_similarity(vec1, vec2)
                if similarity >= semantic_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_items.append(item1)

        return unique_items
    
    def apply_constraints(self, items, category):
        """
        Apply constraints and business logic to the list of items 
        based on the previous item mix and other constraints.

        Args:
            items: The list of items to apply constraints to.
            category: The category of items.

        Returns:
            A list of items satisfying the constraints.
        """
        # semantic filtering to prevent near-duplicate items
        semantic_constrained_items1 = self._apply_semantic_constraints(items)  # Levenstein distances
        semantic_constrained_items2 = self._apply_semantic_constraints_using_embeddings(semantic_constrained_items1)  # embeddings and cosine similarity

        # ensure core-products weren't removed
        missing_core = [x for x in self.core_items if x not in semantic_constrained_items2]
        if len(missing_core) > 0:
            print(f'Core products were removed after item-de-duplication, and are added back: {[x.name for x in missing_core]}')
            constrained_items = missing_core + semantic_constrained_items2
        else:
            constrained_items = semantic_constrained_items2

        # price constraint
        price_constrained_items = [item for item in constrained_items if item.get_price() <= self.max_price]
        excluded_items = set(items) - set(price_constrained_items)

        # previous item constraint
        # core-products are not considered previous_items for this logic
        previous_items = [x for x in self.previous_items.get(category, []) if x not in self.core_items]
        ranked_previous_items = sorted(previous_items, key=lambda x: x.reward_function(), reverse=True)
        other_previous_items = ranked_previous_items[self.num_previous_items:]
        # Exclude all previous items that are not top-performing
        constrained_items = [item for item in price_constrained_items if item not in other_previous_items]

        # If increase_repeats is True, add back items from other previous_items
        if self.increase_repeats and len(constrained_items) < self.num_items[category]:
            possible_additions = set(other_previous_items) - excluded_items
            for item in possible_additions:
                constrained_items.append(item)
                if len(constrained_items) == self.num_items[category]:
                    break

        # warning outputs
        if len(set(constrained_items).intersection(excluded_items)) != 0:
            print('Excluded items have been selected in the item mix')
        if len(set(constrained_items).intersection(set(self.core_items))) != len(self.core_items):
            missing_core = set(self.core_items) - (set(constrained_items).intersection(set(self.core_items)))
            print(f'Some detected core products were not included in the item mix: {missing_core}')
        
        self.chosen_items = constrained_items

        if self.verbose:
            print('\n', category)
            print('Core Products detected: ', len(self.core_items))
            print('Core Products: ', [x.name for x in self.core_items])
            print('Items before semantic filter 1: ', len(items))
            print('Items after semantic filter 1: ', len(semantic_constrained_items1))
            print('Items after semantic filter 2: ', len(semantic_constrained_items2))
            print(len(excluded_items), [x.name for x in excluded_items])

            print('Number of desired items: ', self.num_items[category])
            print('Number of selected items: ', len(constrained_items))

            if len(excluded_items) > 0:
                print('Selected Items: ', [x.name for x in constrained_items])

        return constrained_items
