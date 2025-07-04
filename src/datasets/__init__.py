from .yelp import YelpDataset
from .retail import RetailDataset
from .ijcai import IjcaiDataset

DATASETS = {
    YelpDataset.code(): YelpDataset,
    RetailDataset.code(): RetailDataset,
    IjcaiDataset.code(): IjcaiDataset
}

def dataset_factory(
        dataset_code,
        target_behavior,
        multi_behavior,
        min_uc,
        ):
    dataset = DATASETS[dataset_code]
    return dataset(target_behavior, multi_behavior, min_uc)
