"""UAV-OBB dataset adapter: acquisition, local layout, and YOLO-OBB split discovery."""

from adapters.uav_obb.acquire import (
    KAGGLE_MIRROR,
    MENDELEY_DATASET_ID,
    MENDELEY_DOI,
    MENDELEY_LANDING_URL,
    RemoteFile,
    find_bundled_videos,
    find_dataset_root,
    has_splits,
    install_splits,
    install_videos,
    mendeley_files_url,
    parse_file_listing,
    sha256_of,
)
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
    "KAGGLE_MIRROR",
    "MENDELEY_DATASET_ID",
    "MENDELEY_DOI",
    "MENDELEY_LANDING_URL",
    "ObbDataset",
    "RemoteFile",
    "find_bundled_videos",
    "find_dataset_root",
    "has_splits",
    "install_splits",
    "install_videos",
    "is_stabilized_name",
    "iter_split_pairs",
    "list_raw_videos",
    "mendeley_files_url",
    "parse_file_listing",
    "prefer_stabilized",
    "repo_relative",
    "sha256_of",
]
