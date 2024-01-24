from pandera.typing import Series, String, Float, Int
import pandera as pa
from typing import TypedDict, Dict

class ItemDataFrameSchema(pa.SchemaModel):
    Location: Series[String]
    ItemName: Series[String]
    Price: Series[Float]
    SELL_THROUGH_RATE: Series[Float]
    Category: Series[String]

class SimulationResultBaseSchema(pa.SchemaModel):
    Category: Series[String]
    SelectedItem: Series[String]
    Price: Series[Float]
    ItemReward: Series[Float]
    CategoryReward: Series[Float]
    simulation: Series[Int]

class SalesHistorySchema(pa.SchemaModel):
    Date: Series[String]
    LevelDetails: Series[String]
    Sold: Series[Float]
    Level: Series[String]

class AggregatedSalesHistorySchema(pa.SchemaModel):
    YearWeek: Series[String]
    Date: Series[String]
    Price: Series[Float]
    TOTAL_DELIVERED: Series[Float]
    TOTAL_SOLD: Series[Float]
    SELL_THROUGH_RATE: Series[list]

class SelectedItemsSchema(pa.SchemaModel):
    Category: Series[String]
    SelectedItem: Series[String]
    Price: Series[Float]
    ItemReward: Series[Float]
    CategoryReward: Series[Float]

class SelectedItemsLocationsSchema(pa.SchemaModel):
    Location: Series[String]
    Category: Series[String]
    SelectedItem: Series[String]
    Price: Series[Float]
    ItemReward: Series[Float]
    CategoryReward: Series[Float]
    LocationReward: Series[String]

class LocationConstraint(TypedDict):
    n_items: Dict[str, int]
    max_price: float
    n_previous_items: int
    increase_repeats: bool

class PreviousItemHistorySchema(pa.SchemaModel):
    Location: Series[String]
    ItemName: Series[String]
