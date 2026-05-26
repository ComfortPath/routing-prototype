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
    MAX_ROUTE_STATE = 30

    def __init__(
        self,
        schema: NetworkSchema,
        directed: bool = False,
    ) -> None:
        # save original information so the actual route can be retrieved when it's calculated on the NumpyRoutingNetwork
        self.metadata = dict(schema.metadata)
        self.edges = schema.edges.reset_index(drop=True)
        self.directed = directed
        self.edge_length = self.edges["length"].to_numpy(dtype=float)

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
        self.edge_cost = self.edge_length

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

    def extract_edge_values(
        self,
        column: str,
        hour: int | None = None,
        ) -> np.ndarray:
        """
        Extract one numeric value per edge.

        Supports:
        - scalar edge columns, e.g. length, slope, shade_score
        - hourly/list-like edge columns, e.g. utci_category = [5, 5, 6, 7, ...]
        """
        # get all values from the column
        raw_values = self.edges[column].to_numpy()
        values = np.zeros(len(raw_values), dtype=float)

        for i, value in enumerate(raw_values):
            arr = np.asarray(value, dtype=float).reshape(-1)

            # if you choose a variable with a single value (e.g. length)
            if arr.size == 1:
                values[i] = float(arr[0])
            else:
                # get the correct hour when values are saved as arrays.
                values[i] = float(arr[hour])

        return values

    def add_weight(
        self,
        edge_idx: int,
        weight_config: dict[str, Any],
        route_state: int,
    ) -> tuple[float, int]:
        """
        Update the cost of one edge to include environmental information,
        as given by weight_config.

        The edge cost is:

            C_e = L_e * (1 + P_e)

        where:

            P_e = sum_i w_i * gamma_i(s) * p_i

        with:
            w_i:
                User-defined importance of variable i.

            gamma_i(s):
                Dynamic sensitivity function that depends on route state s.
                Here, s is the current heat-exposure state of the route.

            p_i:
                Normalized environmental penalty on edge e.

        The route state increases on hot edges and recovers on non-hot edges.
        """
        length = float(self.edge_length[edge_idx])

        thermal_multiplier = {
            5: 0.0,   # no thermal stress
            6: 0.10,  # moderate heat
            7: 0.35,  # strong heat
            8: 0.90,  # very strong heat
            9: 1.50,  # extreme heat / near-avoid
        }

        total_penalty = 0.0
        edge_is_hot = False

        for variable in weight_config["variables"]:
            name = variable["name"]
            w_i = float(variable["w"])

            gamma = variable.get("gamma", 1.0)
            if callable(gamma):
                gamma_i = float(gamma(route_state))
            else:
                gamma_i = float(gamma)

            edge_value = variable["_values"][edge_idx]

            if name == "utci_category":
                utci_category = int(edge_value)
                p_i = thermal_multiplier[utci_category]

                hot_categories = variable.get("hot_categories", [7, 8, 9])
                if utci_category in hot_categories:
                    edge_is_hot = True

            else:
                p_i = float(edge_value)

                if variable.get("counts_as_hot", False):
                    hot_threshold = variable.get("hot_threshold", 0.0)
                    if p_i >= hot_threshold:
                        edge_is_hot = True

            total_penalty += w_i * gamma_i * p_i

        edge_cost = length * (1.0 + total_penalty)

        hot_state_increment = weight_config.get("hot_state_increment", 3)
        cold_state_recovery = weight_config.get("cold_state_recovery", 1)

        if edge_is_hot:
            new_route_state = route_state + hot_state_increment
        else:
            new_route_state = route_state - cold_state_recovery

        new_route_state = max(0, min(new_route_state, self.MAX_ROUTE_STATE))

        return edge_cost, new_route_state


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
        weight_config: dict[str, Any],
    ) -> np.ndarray | None:
        """
        Weighted Dijkstra implementation with dynamic route-state costs.

        Adjusted from:
        https://gist.github.com/potpath/b1cc6383e1116e895ac2ec891f666888

        Because edge costs can depend on s, the algorithm keeps track of labels as:

            (node, route_state)

        instead of only:

            node

        Returns
        -------
        np.ndarray | None
            The route as original node IDs, or None if no path is found.
        """
        source = self.node_to_idx[source_node_id]
        target = self.node_to_idx[target_node_id]

        # Copy the config so the original dictionary is not modified.
        routing_config = {
            **weight_config,
            "variables": [dict(variable) for variable in weight_config["variables"]],
        }

        # Extract values for the given variables
        for variable in routing_config["variables"]:
            variable["_values"] = self.extract_edge_values(
                column=variable["name"],
                hour=variable.get("hour"),
            )

        initial_state = 0
        # keep track of how many hot edges were visited before
        start_label = (source, initial_state)

        # Keep track of shortest cost to each visited node-state combination.
        visited = {
            start_label: 0.0
        }

        # Priority queue using heapq, containing:
        # (current route cost, current node, current route state)
        h = [
            (0.0, source, initial_state)
        ]

        # Keep track of previous node-state combinations.
        path = {}

        # Keep all processed node-state combinations.
        processed = set()

        final_label = None

        while h:
            current_cost, min_node, route_state = heapq.heappop(h)
            current_label = (min_node, route_state)

            # if already visited
            if current_label in processed:
                continue

            processed.add(current_label)

            # Found path.
            if min_node == target:
                final_label = current_label
                break

            # Loop over all neighbors and save the shortest dynamic-state path.
            for v, edge_idx in self.adjacency[min_node]:
                edge_cost, new_route_state = self.add_weight(
                    edge_idx=edge_idx,
                    weight_config=routing_config,
                    route_state=route_state,
                )

                new_cost = current_cost + edge_cost
                new_label = (v, new_route_state)

                if new_label not in visited or new_cost < visited[new_label]:
                    visited[new_label] = new_cost
                    heapq.heappush(h, (new_cost, v, new_route_state))
                    path[new_label] = current_label

        if final_label is None:
            return None

        route = []
        label = final_label

        # Reconstruct the original path through its predecessor labels.
        while label != start_label:
            node, state = label
            route.append(node)
            label = path[label]

        route.append(source)
        route = np.array(route[::-1], dtype=np.int64)

        return self.node_ids[route]
    
if __name__ == "__main__":
    schema = NetworkSchema.from_folder("data/network_final/red_bbox")
    routing_network = NumpyRoutingNetwork(schema)
    weight_config = {
        "variables": [
            {
                "name": "utci_category",
                "w": 1.0,
                "hour": 14,
                "gamma": lambda s: 1.0 + 0.05 * s,
                "hot_categories": [7, 8, 9],
            }
        ]
    }
    route = routing_network.shortest_path(
        6015228571,
        12697345954,
        weight_config=weight_config,
    )
    