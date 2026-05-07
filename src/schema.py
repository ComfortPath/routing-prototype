from __future__ import annotations

import heapq
import logging
import math
from pathlib import Path
import json

from collections.abc import Callable
from typing import Any

import numpy as np
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import networkx as nx

# ---------------------------------------------------------------------------
# Internal representation of the network
# ---------------------------------------------------------------------------
@dataclass
class NetworkSchema:
    """
    Internal persistence for the routing network, same as present in the network builder repository

    This class is only used in this file and it's main purpose is to generalize the distribution
    of the network to a predefined schema.

    The schema can be published as folder:
    network/
    ├── nodes.parquet
    ├── edges.parquet
    └── metadata.json
    """
    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    metadata: Dict[str, Any]

    def validate(self) -> None:
        """
        Check if all necessary columns to reconstruct a graph are present.
        """

        required_node_cols = {"node_id", "geometry"}
        required_edge_cols = {"u", "v", "key", "geometry"}

        missing_node_cols = required_node_cols - set(self.nodes.columns)
        missing_edge_cols = required_edge_cols - set(self.edges.columns)

        if missing_node_cols:
            raise ValueError(
                f"Missing required node columns: {sorted(missing_node_cols)}"
            )
        if missing_edge_cols:
            raise ValueError(
                f"Missing required edge columns: {sorted(missing_edge_cols)}"
            )
        
    @classmethod
    def from_folder(cls, folder: str | Path) -> "NetworkSchema":
        """
        Load nodes, edges and metadata from a network schema folder.
        """

        folder = Path(folder)

        nodes_path = folder / "nodes.parquet"
        edges_path = folder / "edges.parquet"
        metadata_path = folder / "metadata.json"

        nodes = gpd.read_parquet(nodes_path)
        edges = gpd.read_parquet(edges_path)

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        schema = cls(
            nodes=nodes,
            edges=edges,
            metadata=metadata,
        )

        schema.validate()

        return schema
    