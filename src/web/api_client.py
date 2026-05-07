"""Small async client for the pedestrian routing FastAPI server."""

from __future__ import annotations

import httpx


API_BASE_URL = "http://127.0.0.1:8001"


async def fetch_network() -> dict:
    """Fetch the full network payload from GET /route/network."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/route/network")
        response.raise_for_status()
        return response.json()


async def fetch_route(
    coord_a: tuple[float, float],
    coord_b: tuple[float, float],
    hour: int,
) -> dict:
    """Request a UTCI-aware route between two lon/lat coordinates."""
    payload = {
        "origin": {"lon": coord_a[0], "lat": coord_a[1]},
        "destination": {"lon": coord_b[0], "lat": coord_b[1]},
        "time": {"hour": int(hour)},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/route/path", json=payload)
        response.raise_for_status()
        return response.json()
