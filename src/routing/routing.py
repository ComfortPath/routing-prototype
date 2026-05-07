from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from ..schema import NetworkSchema


class NumpyRoutingNetwork:
    """
    Routing representation based on NumPy arrays and an adjacency list.

    Uses the NetworkSchema defined in the parent folder and converts the geoparquet files (which were built according to this schema),
    to a network representation that is suitable for routing.

    """

    def __init__(
        self,
        schema: NetworkSchema,
        directed: bool = False,
    ) -> None:
        # save original information so the actual route can be retrieved when it's calculated on the NumpyRoutingNetwork
        self.metadata = dict(schema.metadata)
        self.edges = schema.edges.reset_index(drop=True)
        self.directed = directed

        # keep originals OSM ids
        self.node_ids = (
            schema.nodes["node_id"].to_numpy()
            if "node_id" in schema.nodes.columns
            else schema.nodes.index.to_numpy()
        )

        # save internal idx suitable to use as an numpy index and create lookup table that maps the new idx to the osm_id
        # idx are dense: 0 .. node.size, while previous osm ids are 1827394029 or 187283749, which would require A LOT of empty rows
        self.node_to_idx = {
            node_id: idx
            for idx, node_id in enumerate(self.node_ids)
        }

        # adjust original edge_ids to correspond to the internal ix
        self.edge_u = np.array(
            [self.node_to_idx[u] for u in self.edges["u"]],
            dtype=np.int64,
        )
        self.edge_v = np.array(
            [self.node_to_idx[v] for v in self.edges["v"]],
            dtype=np.int64,
        )

        # update edge cost to the length of the edge
        self.edge_cost = self.edges["length"].to_numpy(dtype=float)

        # get size of arrays
        self.n_nodes = len(self.node_ids)
        self.n_edges = len(self.edges)

        # create empty adjacency list
        self.adjacency: list[list[tuple[int, int]]] = [
            [] for _ in range(self.n_nodes)
        ]

        # fill adjacency list with all the connected nodes
        for edge_idx, (u, v) in enumerate(zip(self.edge_u, self.edge_v)):
            self.adjacency[u].append((v, edge_idx))

            # if input graph is undirected add both directions as possibilities for routing
            # u -> v and v -> u
            if not directed:
                self.adjacency[v].append((u, edge_idx))

    def add_weights(self, columns: list[str], length: np.ndarray, hour: int) -> np.ndarray:
        """
        Update the edge weight to also include the environmental information

        with C_e = L_e * (1 + P_e). The cost of an edge is dependent on the lenght plus the environemental penalty.

        Penalty is decided by P_e = sum_(i=1)^n w_i * gamma_i (s) * p_i
        with:
        $w$: being the user defined importance of a variable 
        $gamma$: being the dynamic sensitivity function that depends on route-state $s$.
        $p$: the normalized envioronmental penalty on edge $e$
        """
        thermal_multiplier = {
            5: 0.0,  # no thermal stress
            6: 0.10,  # moderate heat
            7: 0.35,  # strong heat
            8: 1.00,  # very strong heat
            9: 3.00,  # extreme heat / near-avoid
        }
        total_penalty = np.zeros_like(length, dtype=float)

        for attribute in columns:
            penalty = self.edges[attribute].to_numpy()
            mapped_penalty = np.array([
                thermal_multiplier[int(p[hour])] for p in penalty
            ])

            total_penalty += mapped_penalty

        weights = length * (1.0 + total_penalty)
        return weights


    def neighbors(self, node_id: Any) -> list[tuple[Any, int, float]]:
        """
        Given a original node, use the adjacency list to return it's neighboring 
        nodes and the edge cost to get there

        Return neighbors as:
            (neighbor_node_id, edge_row, edge_cost)
        
        """
        node_idx = self.node_to_idx[node_id]

        return [
            (self.node_ids[neighbor_idx], edge_idx, self.edge_cost[edge_idx])
            for neighbor_idx, edge_idx in self.adjacency[node_idx]
        ]

    def shortest_path(
        self,
        source_node_id: Any,
        target_node_id: Any,
        hour: int,
    ) -> np.ndarray | None:
        """
         Basic weighted dijkstra implementation.
         Adjusted from https://gist.github.com/potpath/b1cc6383e1116e895ac2ec891f666888
        """

        source = self.node_to_idx[source_node_id]
        target = self.node_to_idx[target_node_id]

        edge_weights = self.add_weights(["utci_category"], self.edge_cost, hour)
        # keep track of shortest distance to all visited edges
        visited = {source: 0.0}

        # priority queue using heapq, which the node and it's associated weight/distance
        h = [(0.0, source)]

        # keep track of all the previous nodes 
        path = {}

        # keep all the nodes that haven't been considered
        nodes = set(range(self.n_nodes))

        # continue while there are still nodes to reach, dijkstra
        while nodes and h:
            current_weight, min_node = heapq.heappop(h)
            try:
                while min_node not in nodes:
                    current_weight, min_node = heapq.heappop(h)
            except IndexError:
                break

            # min node has be processed
            nodes.remove(min_node)

            # found path!
            if min_node == target:
                break
            
            # loop over all the neighbors of the current node and save the shortest path
            for v, edge_idx in self.adjacency[min_node]:
                weight = current_weight + edge_weights[edge_idx]

                if v not in visited or weight < visited[v]:
                    visited[v] = weight
                    heapq.heappush(h, (weight, v))
                    path[v] = min_node

        if target not in visited:
            return None

        route = [target]
        # reconstruct the original path through it's predecessor
        while route[-1] != source:
            route.append(path[route[-1]])

        route = np.array(route[::-1], dtype=np.int64)

        return self.node_ids[route]
    
if __name__ == "__main__":
    schema = NetworkSchema.from_folder("data/big_network")
    routing_network = NumpyRoutingNetwork(schema)
    print(routing_network.n_nodes)
    print(routing_network.n_edges)