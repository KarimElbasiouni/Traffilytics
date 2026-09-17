"""UAV-OBB dataset adapter: local layout and YOLO-OBB split discovery."""

from adapters.uav_obb.dataset import ObbDataset, iter_split_pairs
from adapters.uav_obb.layout import (
    DatasetLayout,
    is_stabilized_name,
    list_raw_videos,
    prefer_stabilized,
    repo_relative,
)

__all__ = [
    "DatasetLayout",
    "ObbDataset",
    "is_stabilized_name",
    "iter_split_pairs",
    "list_raw_videos",
    "prefer_stabilized",
    "repo_relative",
]
