from typing import Optional

from pydantic import BaseModel, Field  # , field_validator


class PartitionConfig(BaseModel):
    split_type_clusters: Optional[str] = None
    split_type_nodes: Optional[str] = None
    num_classes: int = Field(...)
    num_nodes: Optional[int] = None
    num_clusters: Optional[int] = None
    alpha: Optional[int] = None
    percentage_configuration: Optional[dict] = None
    store_path: str = Field(...)

    # @field_validator("num_nodes", "num_clusters", mode="before")
    # def check_none(cls, num_nodes: int | None, values: dict[str, int | None]) -> int:
    #     num_clusters = values.get("num_clusters")
    #     if num_nodes is None and num_clusters is None:
    #         raise ValueError('one of "num_nodes" or "num_clusters" needs to be set')
    #     return num_nodes


class Preferences(BaseModel):
    dataset: str = Field(...)
    data_split_config: PartitionConfig
