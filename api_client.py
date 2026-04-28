import httpx


API_BASE_URL = "http://127.0.0.1:8001"


async def fetch_network() -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/route/network")
        response.raise_for_status()
        return response.json()


async def fetch_route(
    coord_a: tuple[float, float],
    coord_b: tuple[float, float],
    hour: int,
) -> dict:
    """POST two [lon, lat] coordinates and return the route payload.

    Expected response shape (adapt to your API):
        {
            "geojson": { ...FeatureCollection with route geometry... },
            "distance_m": 1234.5,
            "duration_s": 456.7,
        }
    """
    payload = {
        "origin": {"lon": coord_a[0], "lat": coord_a[1]},
        "destination": {"lon": coord_b[0], "lat": coord_b[1]},
        "time": {"hour": int(hour)}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/route/path", json=payload)
        response.raise_for_status()
        return response.json()
