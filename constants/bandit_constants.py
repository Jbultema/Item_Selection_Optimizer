## bandit_item.py defaults
DEFAULT_ITEM_PRICE = 5.0
DEFAULT_SELL_THROUGH_RATE = [0.0]
DEFAULT_DISCOUNT_FACTOR = 0.9
DEFAULT_TIMEOUT_DURATION = 2
DEFAULT_BOOTSTRP_OPTION = True

## bandit_manager.py defaults
DEFAULT_EPSILON = 0.5
DEFAULT_EPSILON_DECAY = 0.1
DEFAULT_MIN_EPSILON = 0.1
DEFAULT_NUM_ITEMS = 5
DEFAULT_MAX_PRICE = 29.99
DEFAULT_NUM_PREVIOUS = 1
DEFAULT_INCREASE_REPEATS = True
DEFAULT_NUM_CATEGORY_ITEMS = 3
DEFAULT_PROTECT_CORE_PRODUCTS = True
DEFAULT_MAX_CORE_PRODUCTS = 10  # total number of possible core-products in a category
DEFAULT_DETECT_CORE_PRODUCT_SENSITIVITY = 0.5  # 0 allows most core-products, 1 allows the least core-products
DEFAULT_LOOKBACK_WINDOW = 42  # number of days in history for core-product analysis
DEFAULT_NUM_ITEM_BUFFER_FROM_CORE = 1  # number of item-slots for a category that cannot be core items
DEFAULT_MAX_LEVENSHTEIN_DISTANCE_SAME_ITEM = 5  # number of characters in string-1 that need to changed to match string-2
DEFAULT_MIN_COSINESIMILARITY_SAME_ITEM = 0.90  # cosine similarity of 0.9 or greater characterizes two items as the same (0.89 would be two different items)

## data_prep.py defaults
DEFAULT_HISTORY_LENGTH = 56
DEFAULT_NUM_OBSERVATIONS = 20
DEFAULT_MAX_SELL_THROUGH = 1
