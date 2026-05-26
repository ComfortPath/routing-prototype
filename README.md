# Thermal Pedestrian Routing Prototype

This repository contains a prototype application for calculating and visualising thermally aware pedestrian routes. The application combines a FastAPI routing backend with a web-based MapLibre/Shiny interface. Users can select an origin and destination on the map, choose an hour of the day, and calculate a route that accounts for hourly UTCI-based thermal conditions.

## Code overview

### `src/routing`

The `routing` module contains the backend logic for loading the pedestrian network and calculating routes.

- `main.py` defines the FastAPI routing server. It loads the prepared network data, exposes the network as GeoJSON, receives route requests, snaps origin and destination coordinates to the nearest network nodes, and returns calculated route geometries and summary statistics.
- `routing.py` contains the NumPy-based routing implementation. It represents the network using arrays and adjacency lists and computes shortest paths with support for dynamic UTCI-based edge weighting.

### `src/web`

The `web` module contains the user-facing web application.

- `app.py` defines the Shiny application layout and sidebar controls.
- `server_ui.py` contains the server-side UI logic, including map rendering, network colouring, marker handling, route requests, and displayed route statistics.
- `api_client.py` is a small asynchronous client used by the web application to communicate with the FastAPI routing backend.

### `src/schema.py`

`schema.py` defines the shared network schema used to read and write the prepared network data. The routing backend expects the network to follow this schema so that nodes, edges, geometry, metadata, and routing attributes are consistently available.

## Starting the application

The application consists of two servers:

1. the routing backend, running on port `8001`;
2. the web application, running on port `8000`.

Start both servers from the project root in separate terminal windows.

### Install the dependencies
To run the servers first install the repository dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
### 1. Start the routing backend

```bash
uvicorn src.routing.main:app --host 0.0.0.0 --port 8001
```

The routing backend exposes endpoints for retrieving the network and calculating routes.

The API endpoints:

```text
http://127.0.0.1:8001/health
http://127.0.0.1:8001/route/network
http://127.0.0.1:8001/route/path
```

The `/health` endpoint can be used to check whether the network has been loaded successfully.

### 2. Start the web application

```bash
uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

Then open the web application in a browser:

```text
http://127.0.0.1:8000
```

The web application expects the routing backend to be available at:

```text
http://127.0.0.1:8001
```

The web application and routing backend must both be running for the application to work.
