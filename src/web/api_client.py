"""Async client for the pedestrian routing FastAPI server."""

from __future__ import annotations

from typing import Any

import httpx


API_BASE_URL = "http://127.0.0.1:8001"


async def fetch_network() -> dict[str, Any]:
    """Fetch the full network GeoJSON payload from the routing API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/route/network")
        response.raise_for_status()
        return response.json()


async def fetch_route(
    coord_a: tuple[float, float],
    coord_b: tuple[float, float],
    hour: int,
    weight_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request a weighted route between two longitude/latitude coordinates."""
    payload: dict[str, Any] = {
        "origin": {"lon": coord_a[0], "lat": coord_a[1]},
        "destination": {"lon": coord_b[0], "lat": coord_b[1]},
        "time": {"hour": int(hour)},
    }

    if weight_config is not None:
        payload["weight_config"] = weight_config

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/route/path", json=payload)
        response.raise_for_status()
        return response.json()
