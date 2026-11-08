from __future__ import annotations

import heapq
import math
import re
import time
from collections import defaultdict
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN
from ipyleaflet import (
    CircleMarker,
    GeoJSON,
    LegendControl,
    Map,
    Polyline,
    TileLayer,
)
from ipywidgets import HTML
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget

from datetime import date

# -----------------------------------------------------------------------------
# Paths and data loading
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".cache"
ONE_WEEK_SECONDS = 7 * 24 * 60 * 60


def find_local_file(*names: str) -> Path:
    """Return the first existing local filename, with a helpful fallback path."""
    for filename in names:
        candidate = APP_DIR / filename
        if candidate.exists():
            return candidate
    return APP_DIR / names[0]


ADDRESS_CSV = find_local_file("Parcel Address.csv")
STREET_CSV = find_local_file("Street Centreline.csv")
PLUS15_CSV = find_local_file("plus15.csv", "plus15(1).csv")

DATASET_URLS = {
    # These datasets feed the pre-existing transit dashboard.
    # All Plus 15 / address / street-routing data are loaded from local CSVs.
    "ridership": "https://data.calgary.ca/resource/iema-jbc4.json",
    "ctrain": "https://data.calgary.ca/resource/3u7d-jj5c.json",
    "budget": "https://data.calgary.ca/resource/fqax-i3nz.json",
}


def load_dataset(
    name: str, url: str, max_age_seconds: int = ONE_WEEK_SECONDS
) -> pd.DataFrame:
    """Load an existing transit dashboard dataset from local CSV/cache/API."""
    local_file = APP_DIR / f"{name}.csv"
    if local_file.exists():
        try:
            return pd.read_csv(local_file)
        except Exception as exc:
            print(f"[{name}] Could not read {local_file}: {exc}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{name}.csv"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < max_age_seconds:
            try:
                return pd.read_csv(cache_file)
            except Exception as exc:
                print(f"[{name}] Could not read cached file: {exc}")

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df.to_csv(cache_file, index=False)
        return df
    except Exception as exc:
        print(f"[{name}] API fetch failed: {exc}")
        if cache_file.exists():
            try:
                return pd.read_csv(cache_file)
            except Exception:
                pass
        return pd.DataFrame()


for required_file in (ADDRESS_CSV, STREET_CSV, PLUS15_CSV):
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required local CSV not found: {required_file}\n"
            "Place the CSV beside app.py. Expected files: Parcel Address.csv, "
            "Street Centreline.csv, and plus15.csv."
        )

# Load routing data locally only. No API calls are made for these three files.
address_df = pd.read_csv(
    ADDRESS_CSV,
    usecols=[
        "ADDRESS",
        "STREET_NAME",
        "STREET_TYPE",
        "STREET_QUAD",
        "HOUSE_NUMBER",
        "HOUSE_ALPHA",
        "longitude",
        "latitude",
    ],
    low_memory=False,
)
plus15_df = pd.read_csv(PLUS15_CSV, low_memory=False)
street_df = pd.read_csv(
    STREET_CSV,
    usecols=["BUILT_STATUS", "ONE_WAY", "multiline"],
    low_memory=False,
)

# Clean searchable address columns once at startup.
for col in ("STREET_NAME", "STREET_TYPE", "STREET_QUAD", "HOUSE_ALPHA"):
    address_df[col] = address_df[col].fillna("").astype(str).str.strip().str.upper()
address_df["HOUSE_NUMBER"] = pd.to_numeric(address_df["HOUSE_NUMBER"], errors="coerce")
address_df["longitude"] = pd.to_numeric(address_df["longitude"], errors="coerce")
address_df["latitude"] = pd.to_numeric(address_df["latitude"], errors="coerce")
address_df = address_df.dropna(subset=["HOUSE_NUMBER", "longitude", "latitude"]).copy()
address_df["HOUSE_NUMBER"] = address_df["HOUSE_NUMBER"].astype(int)
address_df["ADDRESS"] = address_df["ADDRESS"].fillna("").astype(str).str.strip()


# -----------------------------------------------------------------------------
# Geometry and address helpers
# -----------------------------------------------------------------------------
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COORD_RE = re.compile(NUMBER_RE)
WKT_PART_RE = re.compile(r"\(([^()]*)\)")


def parse_wkt_parts(wkt_value: object) -> list[list[tuple[float, float]]]:
    """Parse 2D WKT coordinate groups into lists of (longitude, latitude)."""
    if not isinstance(wkt_value, str) or not wkt_value.strip():
        return []
    parts: list[list[tuple[float, float]]] = []
    for part in WKT_PART_RE.findall(wkt_value):
        values = COORD_RE.findall(part)
        if len(values) < 4 or len(values) % 2:
            continue
        coords = list(zip(map(float, values[0::2]), map(float, values[1::2])))
        if len(coords) >= 2:
            parts.append(coords)
    return parts


def parse_wkt_coordinates(wkt_value: object) -> list[tuple[float, float]]:
    """Flatten WKT parts for the simple polygon geometries in plus15.csv."""
    parts = parse_wkt_parts(wkt_value)
    return parts[0] if parts else []


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate local Calgary distance between (lon, lat) points."""
    return math.hypot((b[1] - a[1]) * 111_139, (b[0] - a[0]) * 70_000)


STREET_TYPE_ALIASES = {
    "AVENUE": "AV",
    "AVE": "AV",
    "AV": "AV",
    "STREET": "ST",
    "ST": "ST",
    "ROAD": "RD",
    "RD": "RD",
    "DRIVE": "DR",
    "DR": "DR",
    "BOULEVARD": "BV",
    "BLVD": "BV",
    "BV": "BV",
    "CRESCENT": "CR",
    "CRES": "CR",
    "CR": "CR",
    "COURT": "CT",
    "CT": "CT",
    "PLACE": "PL",
    "PL": "PL",
    "CLOSE": "CL",
    "CL": "CL",
    "LANE": "LN",
    "LN": "LN",
    "WAY": "WY",
    "WY": "WY",
    "TRAIL": "TR",
    "TR": "TR",
    "TERRACE": "TC",
    "TC": "TC",
    "PARKWAY": "PY",
    "PKWY": "PY",
    "PY": "PY",
    "CIRCLE": "CI",
    "CIR": "CI",
    "CI": "CI",
    "LINK": "LI",
    "LI": "LI",
    "GATE": "GA",
    "GA": "GA",
    "GREEN": "GR",
    "GR": "GR",
    "VIEW": "VW",
    "VW": "VW",
    "WALK": "WK",
    "WK": "WK",
    "RISE": "RI",
    "RI": "RI",
    "SQUARE": "SQ",
    "SQ": "SQ",
    "COMMON": "CM",
    "CM": "CM",
    "GROVE": "GV",
    "GV": "GV",
    "HEATH": "HE",
    "HE": "HE",
    "HEIGHTS": "HT",
    "HT": "HT",
    "ISLAND": "IS",
    "IS": "IS",
    "MOUNT": "MT",
    "MT": "MT",
    "PARADE": "PR",
    "PR": "PR",
    "PASSAGE": "PS",
    "PS": "PS",
    "POINT": "PT",
    "PT": "PT",
    "VILLAS": "VI",
    "VI": "VI",
    "MEWS": "ME",
    "ME": "ME",
    "MANOR": "MR",
    "MR": "MR",
    "COVE": "CV",
    "CV": "CV",
    "BAY": "BA",
    "BA": "BA",
    "CENTRE": "CE",
    "CE": "CE",
    "ALLEY": "AL",
    "AL": "AL",
}

WORD_ORDINALS = {
    "FIRST": "1",
    "1ST": "1",
    "SECOND": "2",
    "2ND": "2",
    "THIRD": "3",
    "3RD": "3",
    "FOURTH": "4",
    "4TH": "4",
    "FIFTH": "5",
    "5TH": "5",
    "SIXTH": "6",
    "6TH": "6",
    "SEVENTH": "7",
    "7TH": "7",
    "EIGHTH": "8",
    "8TH": "8",
    "NINTH": "9",
    "9TH": "9",
    "TENTH": "10",
    "10TH": "10",
    "ELEVENTH": "11",
    "11TH": "11",
    "TWELFTH": "12",
    "12TH": "12",
}

TYPE_PATTERN = "|".join(
    sorted(map(re.escape, STREET_TYPE_ALIASES), key=len, reverse=True)
)
ADDRESS_PATTERN = re.compile(
    rf"^\s*(\d+)\s*([A-Z]?)\s+(.+?)\s+({TYPE_PATTERN})(?:\s+(NW|NE|SW|SE))?\s*$",
    flags=re.IGNORECASE,
)


def normalize_building_name(name: object) -> str:
    """Standardize building name variations across plus15.csv and user queries."""
    cleaned = re.sub(r"\s+", " ", str(name or "").strip())
    aliases = {
        "333 fifth av": "333 Fifth Avenue SW",
        "333 5th av": "333 Fifth Avenue SW",
        "333 5th ave": "333 Fifth Avenue SW",
        "333 5th ave sw": "333 Fifth Avenue SW",
        "505 3rd st": "505 Third St",
        "505 3rd st sw": "505 Third St",
        "encore place": "Encor Place",
        "millennium tower": "Millenium Tower",
        "bow valley sqare": "Bow Valley Square",
        "stock exchange place": "Stock Exchange Tower",
    }
    return aliases.get(cleaned.casefold(), cleaned)


def parse_address_query(query: str) -> dict | None:
    """Parse addresses such as '520 10 Ave SW' or '520 10th Ave SW' to Calgary parcel fields."""
    cleaned = re.sub(r",?\s*CALGARY(?:\s*,?\s*AB)?\s*$", "", query.strip(), flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned)
    match = ADDRESS_PATTERN.match(cleaned)
    if not match:
        return None
    house, alpha, street_name, street_type, quadrant = match.groups()
    street_clean = street_name.strip().upper()
    street_clean = WORD_ORDINALS.get(
        street_clean,
        re.sub(r"^(\d+)(?:ST|ND|RD|TH)\b", r"\1", street_clean),
    )
    return {
        "house_number": int(house),
        "house_alpha": (alpha or "").upper(),
        "street_name": street_clean,
        "street_type": STREET_TYPE_ALIASES[street_type.upper()],
        "quadrant": (quadrant or "SW").upper(),
    }


# -----------------------------------------------------------------------------
# Build the downtown combined walking graph from the supplied CSV files
# -----------------------------------------------------------------------------
DOWNTOWN_BOUNDS = (
    -114.105,
    -114.035,
    51.015,
    51.085,
)  # min lon, max lon, min lat, max lat
MAX_BUILDING_TO_STREET_M = 120.0
MAX_ADDRESS_TO_STREET_M = 300.0
OUTDOOR_COST_FACTOR = 2.5
STREET_TRANSITION_PENALTY_M = 120.0


def add_graph_edge(graph: dict, a, b, weight: float) -> None:
    if a == b or not math.isfinite(weight) or weight <= 0:
        return
    graph.setdefault(a, {})[b] = min(weight, graph.get(a, {}).get(b, float("inf")))
    graph.setdefault(b, {})[a] = min(weight, graph.get(b, {}).get(a, float("inf")))


def build_routing_network(roads: pd.DataFrame, plus: pd.DataFrame):
    """Build a downtown pedestrian graph from street and Plus 15 CSV geometry."""
    graph: dict = {}
    road_coords: dict = {}
    plus15_polygon_layers: list[dict] = []
    building_endpoint_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    plus15_links: list[dict] = []

    min_lon, max_lon, min_lat, max_lat = DOWNTOWN_BOUNDS

    for wkt_value in roads["multiline"].dropna().astype(str):
        for part in parse_wkt_parts(wkt_value):
            xs = [point[0] for point in part]
            ys = [point[1] for point in part]
            if (
                max(xs) < min_lon
                or min(xs) > max_lon
                or max(ys) < min_lat
                or min(ys) > max_lat
            ):
                continue

            road_nodes = []
            for lon, lat in part:
                node = ("road", round(lon, 6), round(lat, 6))
                road_coords.setdefault(node, (lon, lat))
                graph.setdefault(node, {})
                road_nodes.append(node)

            for node_a, node_b in zip(road_nodes, road_nodes[1:]):
                coord_a = road_coords[node_a]
                coord_b = road_coords[node_b]
                add_graph_edge(
                    graph,
                    node_a,
                    node_b,
                    distance_m(coord_a, coord_b) * OUTDOOR_COST_FACTOR,
                )

    if not road_coords:
        raise RuntimeError(
            "No downtown street-centreline geometry was found in Street Centreline.csv."
        )

    for _, row in plus.iterrows():
        coords = parse_wkt_coordinates(row.get("polygon"))
        if len(coords) < 4:
            continue

        name_a = normalize_building_name(row.get("connection_1", ""))
        name_b = normalize_building_name(row.get("connection_2", ""))
        if (
            not name_a
            or not name_b
            or name_a.lower() == "nan"
            or name_b.lower() == "nan"
        ):
            continue

        end_a = (
            (coords[0][0] + coords[1][0]) / 2.0,
            (coords[0][1] + coords[1][1]) / 2.0,
        )
        end_b = (
            (coords[2][0] + coords[3][0]) / 2.0,
            (coords[2][1] + coords[3][1]) / 2.0,
        )

        building_endpoint_samples[name_a].append(end_a)
        building_endpoint_samples[name_b].append(end_b)
        plus15_links.append(
            {
                "a": name_a,
                "b": name_b,
                "end_a": end_a,
                "end_b": end_b,
                "polygon": coords,
                "bridge_name": str(row.get("bridge_name", "")),
            }
        )
        plus15_polygon_layers.append(
            {
                "name": f"{name_a} ↔ {name_b}",
                "coords": coords,
                "end_a": end_a,
                "end_b": end_b,
            }
        )

    building_coords = {
        name: (
            float(np.median([point[0] for point in samples])),
            float(np.median([point[1] for point in samples])),
        )
        for name, samples in building_endpoint_samples.items()
    }

    for name in building_coords:
        graph.setdefault(("building", name), {})

    plus15_edge_lookup: dict[frozenset, dict] = {}
    for link in plus15_links:
        node_a = ("building", link["a"])
        node_b = ("building", link["b"])
        corridor_length = distance_m(link["end_a"], link["end_b"])
        add_graph_edge(graph, node_a, node_b, corridor_length)
        plus15_edge_lookup[frozenset((node_a, node_b))] = link

    # Connect each Plus 15 building node to its nearest downtown street node.
    road_node_list = list(road_coords)
    road_xy = np.asarray([road_coords[node] for node in road_node_list], dtype=float)

    def nearest_road_node(lon: float, lat: float):
        dx = (road_xy[:, 0] - lon) * 70_000.0
        dy = (road_xy[:, 1] - lat) * 111_139.0
        squared = dx * dx + dy * dy
        index = int(np.argmin(squared))
        return road_node_list[index], float(math.sqrt(squared[index]))

    building_connector_distances = []
    for name, (lon, lat) in building_coords.items():
        nearest, gap = nearest_road_node(lon, lat)
        building_connector_distances.append(gap)
        if gap <= MAX_BUILDING_TO_STREET_M:
            add_graph_edge(
                graph,
                ("building", name),
                nearest,
                gap * OUTDOOR_COST_FACTOR + STREET_TRANSITION_PENALTY_M,
            )

    print("Downtown routing graph built:")
    print(f"  street nodes: {len(road_coords):,}")
    print(f"  Plus 15 buildings: {len(building_coords):,}")
    print(f"  Plus 15 links: {len(plus15_links):,}")
    if building_connector_distances:
        print(
            "  building-to-street connectors within limit: "
            f"{sum(d <= MAX_BUILDING_TO_STREET_M for d in building_connector_distances):,}/"
            f"{len(building_connector_distances):,}"
        )

    return (
        graph,
        road_coords,
        building_coords,
        plus15_polygon_layers,
        plus15_edge_lookup,
        nearest_road_node,
    )


(
    ROUTING_GRAPH,
    ROAD_COORDS,
    BUILDING_COORDS,
    PLUS15_POLYGONS,
    PLUS15_EDGE_LOOKUP,
    nearest_road_node,
) = build_routing_network(street_df, plus15_df)


# -----------------------------------------------------------------------------
# Pre-build lightweight GeoJSON layers for all 104 links & 102 building hubs
# -----------------------------------------------------------------------------
def make_circle_ring_lonlat(
    lon: float, lat: float, radius_m: float = 8.5, sides: int = 14
) -> list[list[float]]:
    """Create a closed GeoJSON polygon ring [lon, lat] around a building hub."""
    dlon = radius_m / 70_000.0
    dlat = radius_m / 111_139.0
    ring = [
        [
            round(lon + dlon * math.cos(2.0 * math.pi * i / sides), 7),
            round(lat + dlat * math.sin(2.0 * math.pi * i / sides), 7),
        ]
        for i in range(sides)
    ]
    ring.append(ring[0])
    return ring


def build_base_geojson_layers():
    skywalk_features = []
    for item in PLUS15_POLYGONS:
        ring = [[lon, lat] for lon, lat in item["coords"]]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        skywalk_features.append(
            {
                "type": "Feature",
                "properties": {"name": item["name"]},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
        skywalk_features.append(
            {
                "type": "Feature",
                "properties": {"name": item["name"]},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [item["end_a"][0], item["end_a"][1]],
                        [item["end_b"][0], item["end_b"][1]],
                    ],
                },
            }
        )

    hub_features = []
    for b_name, (lon, lat) in BUILDING_COORDS.items():
        hub_features.append(
            {
                "type": "Feature",
                "properties": {"building": b_name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [make_circle_ring_lonlat(lon, lat, radius_m=8.5)],
                },
            }
        )

    return (
        {"type": "FeatureCollection", "features": skywalk_features},
        {"type": "FeatureCollection", "features": hub_features},
    )


BASE_SKYWALK_GEOJSON, BASE_HUBS_GEOJSON = build_base_geojson_layers()
EMPTY_GEOJSON = {"type": "FeatureCollection", "features": []}


# -----------------------------------------------------------------------------
# Location lookup: Plus 15 building name or local parcel address
# -----------------------------------------------------------------------------
def resolve_building(query: str) -> str | None:
    q = normalize_building_name(query).casefold()
    if not q:
        return None
    names = list(BUILDING_COORDS)
    for name in names:
        if name.casefold() == q:
            return name
    matches = sorted(
        [name for name in names if q in name.casefold()],
        key=len,
    )
    return matches[0] if matches else None


def resolve_address(query: str) -> dict | None:
    parsed = parse_address_query(query)
    if not parsed:
        normalized = re.sub(r"[^A-Z0-9]", "", query.upper())
        candidate = address_df[
            address_df["ADDRESS"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
            == normalized
        ]
        if not candidate.empty:
            row = candidate.iloc[0]
            return {
                "kind": "address",
                "label": row["ADDRESS"],
                "coord": (float(row["longitude"]), float(row["latitude"])),
                "note": "Exact parcel address match.",
            }
        return None

    mask = (
        (address_df["HOUSE_NUMBER"] == parsed["house_number"])
        & (address_df["HOUSE_ALPHA"] == parsed["house_alpha"])
        & (address_df["STREET_NAME"] == parsed["street_name"])
        & (address_df["STREET_TYPE"] == parsed["street_type"])
        & (address_df["STREET_QUAD"] == parsed["quadrant"])
    )
    exact = address_df.loc[mask]
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "kind": "address",
            "label": row["ADDRESS"],
            "coord": (float(row["longitude"]), float(row["latitude"])),
            "note": "Exact parcel address match.",
        }

    same_street = address_df[
        (address_df["STREET_NAME"] == parsed["street_name"])
        & (address_df["STREET_TYPE"] == parsed["street_type"])
        & (address_df["STREET_QUAD"] == parsed["quadrant"])
        & (address_df["HOUSE_ALPHA"] == "")
    ].copy()
    house = parsed["house_number"]
    lower = same_street[same_street["HOUSE_NUMBER"] < house]
    upper = same_street[same_street["HOUSE_NUMBER"] > house]
    if lower.empty or upper.empty:
        return None

    low_number = int(lower["HOUSE_NUMBER"].max())
    high_number = int(upper["HOUSE_NUMBER"].min())
    if high_number - low_number > 100:
        return None

    low_row = lower[lower["HOUSE_NUMBER"] == low_number].iloc[0]
    high_row = upper[upper["HOUSE_NUMBER"] == high_number].iloc[0]
    fraction = (house - low_number) / (high_number - low_number)
    lon = float(
        low_row["longitude"] + (high_row["longitude"] - low_row["longitude"]) * fraction
    )
    lat = float(
        low_row["latitude"] + (high_row["latitude"] - low_row["latitude"]) * fraction
    )
    requested_label = (
        f"{parsed['house_number']} {parsed['street_name']} "
        f"{parsed['street_type']} {parsed['quadrant']}"
    )
    return {
        "kind": "address",
        "label": requested_label,
        "coord": (lon, lat),
        "note": (
            f"Approximate location: no exact parcel record was found. "
            f"Interpolated between {low_row['ADDRESS']} and {high_row['ADDRESS']}."
        ),
    }


def resolve_location(query: str) -> dict | None:
    if not query or not query.strip():
        return None
    building = resolve_building(query)
    if building:
        return {
            "kind": "building",
            "label": building,
            "node": ("building", building),
            "coord": BUILDING_COORDS[building],
            "note": f"Matched to Plus 15 building ({building}).",
        }
    return resolve_address(query)


def shortest_path(start_node, end_node, extra_adjacency: dict):
    """Dijkstra over the prebuilt graph plus temporary address snap edges."""
    if start_node == end_node:
        return [start_node], 0.0

    serial = count()
    queue = [(0.0, next(serial), start_node)]
    distances = {start_node: 0.0}
    previous = {}
    visited = set()

    while queue:
        distance, _, node = heapq.heappop(queue)
        if node in visited:
            continue
        if node == end_node:
            break
        visited.add(node)

        neighbors = ROUTING_GRAPH.get(node, {})
        for neighbor, weight in neighbors.items():
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, next(serial), neighbor))

        for neighbor, weight in extra_adjacency.get(node, {}).items():
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, next(serial), neighbor))

    if end_node not in distances:
        return [], 0.0

    path = [end_node]
    while path[-1] != start_node:
        path.append(previous[path[-1]])
    path.reverse()
    return path, distances[end_node]


def calculate_route(start_text: str, end_text: str) -> dict:
    if not start_text.strip() or not end_text.strip():
        return {
            "ok": False,
            "message": "",
        }

    start = resolve_location(start_text)
    end = resolve_location(end_text)
    if start is None:
        return {
            "ok": False,
            "message": f"Could not find '{start_text}' in the local address or Plus 15 building data. Check the spelling or Calgary quadrant.",
        }
    if end is None:
        return {
            "ok": False,
            "message": f"Could not find '{end_text}' in the local address or Plus 15 building data. Check the spelling or Calgary quadrant.",
        }

    extra_adjacency: dict = defaultdict(dict)
    endpoint_coords: dict = {}

    def graph_node_for(location: dict, role: str):
        if location["kind"] == "building":
            return location["node"], 0.0
        endpoint_node = ("endpoint", role)
        road_node, snap_distance = nearest_road_node(*location["coord"])
        if snap_distance > MAX_ADDRESS_TO_STREET_M:
            return None, snap_distance
        extra_adjacency[endpoint_node][road_node] = snap_distance * OUTDOOR_COST_FACTOR
        extra_adjacency[road_node][endpoint_node] = snap_distance * OUTDOOR_COST_FACTOR
        endpoint_coords[endpoint_node] = location["coord"]
        return endpoint_node, snap_distance

    start_node, start_snap = graph_node_for(start, "start")
    end_node, end_snap = graph_node_for(end, "end")
    if start_node is None:
        return {
            "ok": False,
            "message": f"'{start_text}' is more than {MAX_ADDRESS_TO_STREET_M:.0f} m from the downtown street network in this map.",
        }
    if end_node is None:
        return {
            "ok": False,
            "message": f"'{end_text}' is more than {MAX_ADDRESS_TO_STREET_M:.0f} m from the downtown street network in this map.",
        }

    path, _ = shortest_path(start_node, end_node, extra_adjacency)
    if not path:
        return {
            "ok": False,
            "message": "No connected walking route was found between these locations.",
        }

    def node_lonlat(node):
        if node[0] == "road":
            return ROAD_COORDS[node]
        if node[0] == "building":
            return BUILDING_COORDS[node[1]]
        return endpoint_coords[node]

    true_distance_m = sum(
        distance_m(node_lonlat(a), node_lonlat(b)) for a, b in zip(path, path[1:])
    )

    route_notes = [
        note for note in (start.get("note", ""), end.get("note", "")) if note
    ]
    return {
        "ok": True,
        "message": f"Route found ({start['label']} → {end['label']}): approximately {true_distance_m:.0f} m. "
        + " ".join(route_notes),
        "path": path,
        "distance_m": true_distance_m,
        "start": start,
        "end": end,
        "endpoint_coords": endpoint_coords,
        "start_snap_m": start_snap,
        "end_snap_m": end_snap,
    }


# -----------------------------------------------------------------------------
# Shiny UI
# -----------------------------------------------------------------------------
ui.page_opts(title="Calgary Plus 15 Map & Transit Dashboard", fillable=False)

with ui.navset_tab():
    with ui.nav_panel("Calgary Plus 15 Map"):
        ui.h1("Calgary Plus 15 Map")
        ui.p(
            "Enter an address or a Plus 15 building name, then "
            "select Find Route to highlight the walking path:"
        )

        with ui.layout_column_wrap(width=1 / 2):
            ui.input_text(
                "start_loc",
                "Starting Location",
                placeholder="e.g. Bankers Hall",
            )
            ui.input_text(
                "end_loc",
                "Destination",
                placeholder="e.g. The Bow",
            )

        ui.input_action_button("find_route", "Find Route", class_="btn-primary")
        ui.p(
            """
            Building-to-street links are nearest-
            street approximations, not verified entrance locations. Some plus 15 connections may be closed or innaccessible. This map is for general reference only and is not a substitute for official city maps.
            """
        )

        @reactive.calc
        @reactive.event(input.find_route, ignore_none=False)
        def route_result():
            return calculate_route(input.start_loc() or "", input.end_loc() or "")

        @render.text
        def route_status():
            return route_result()["message"]

        # Create persistent map and consolidated layers ONCE per session.
        # Clicking Find Route mutates the existing route layers in-place rather
        # than destroying and recreating the Map widget.
        dark_basemap = TileLayer(
            url=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            ),
            attribution="Esri, HERE, Garmin, © OpenStreetMap contributors",
            name="Dark Canvas",
        )
        plus15_map_widget = Map(
            basemap=dark_basemap,
            center=(51.0472, -114.0680),
            zoom=15,
            scroll_wheel_zoom=True,
        )

        base_skywalk_layer = GeoJSON(
            data=BASE_SKYWALK_GEOJSON,
            style={
                "color": "#FFD700",
                "fillColor": "#D4AF37",
                "fillOpacity": 0.82,
                "opacity": 0.95,
                "weight": 2.5,
            },
            name="+15 Skywalk Footprints",
        )
        base_hubs_layer = GeoJSON(
            data=BASE_HUBS_GEOJSON,
            style={
                "color": "#FFD700",
                "fillColor": "#111827",
                "fillOpacity": 1.0,
                "opacity": 1.0,
                "weight": 2,
            },
            name="Building Hubs",
        )
        route_polygons_layer = GeoJSON(
            data=EMPTY_GEOJSON,
            style={
                "color": "#1E3A8A",
                "fillColor": "#1E3A8A",
                "fillOpacity": 0.92,
                "opacity": 1.0,
                "weight": 4,
            },
            name="Selected Route Polygons",
        )
        route_line_layer = Polyline(
            locations=[],
            color="#1E3A8A",
            weight=8,
            opacity=1.0,
            fill=False,
        )
        start_marker_layer = CircleMarker(
            location=(51.0472, -114.0680),
            radius=0,
            color="#FFFFFF",
            fill_color="#16A34A",
            fill_opacity=0.0,
            opacity=0.0,
            weight=2,
        )
        end_marker_layer = CircleMarker(
            location=(51.0472, -114.0680),
            radius=0,
            color="#FFFFFF",
            fill_color="#DC2626",
            fill_opacity=0.0,
            opacity=0.0,
            weight=2,
        )

        plus15_map_widget.add_layer(base_skywalk_layer)
        plus15_map_widget.add_layer(base_hubs_layer)
        plus15_map_widget.add_layer(route_polygons_layer)
        plus15_map_widget.add_layer(route_line_layer)
        plus15_map_widget.add_layer(start_marker_layer)
        plus15_map_widget.add_layer(end_marker_layer)
        plus15_map_widget.add_control(
            LegendControl(
                {
                    "+15 Skywalk Link": "#FFD700",
                    "Selected Route": "#1E3A8A",
                    "Start": "#16A34A",
                    "Destination": "#DC2626",
                },
                name="Calgary Walking Network",
                position="bottomright",
            )
        )

        @reactive.effect
        def _update_route_on_map():
            result = route_result()
            if not result.get("ok") or len(result.get("path", [])) < 2:
                route_line_layer.locations = []
                route_polygons_layer.data = EMPTY_GEOJSON
                start_marker_layer.radius = 0
                start_marker_layer.opacity = 0.0
                start_marker_layer.fill_opacity = 0.0
                end_marker_layer.radius = 0
                end_marker_layer.opacity = 0.0
                end_marker_layer.fill_opacity = 0.0
                return

            route_path = result["path"]
            route_positions = []
            highlight_features = []

            for node in route_path:
                if node[0] == "road":
                    lon, lat = ROAD_COORDS[node]
                elif node[0] == "building":
                    lon, lat = BUILDING_COORDS[node[1]]
                    highlight_features.append(
                        {
                            "type": "Feature",
                            "properties": {"building": node[1]},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    make_circle_ring_lonlat(lon, lat, radius_m=11.0)
                                ],
                            },
                        }
                    )
                elif node[0] == "endpoint":
                    lon, lat = result["endpoint_coords"][node]
                else:
                    continue
                route_positions.append((lat, lon))

            for node_a, node_b in zip(route_path, route_path[1:]):
                if node_a[0] == "building" and node_b[0] == "building":
                    link = PLUS15_EDGE_LOOKUP.get(frozenset((node_a, node_b)))
                    if link:
                        ring = [[lon, lat] for lon, lat in link["polygon"]]
                        if ring[0] != ring[-1]:
                            ring.append(ring[0])
                        highlight_features.append(
                            {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {"type": "Polygon", "coordinates": [ring]},
                            }
                        )

            route_line_layer.locations = route_positions
            route_polygons_layer.data = {
                "type": "FeatureCollection",
                "features": highlight_features,
            }

            start_lon, start_lat = result["start"]["coord"]
            end_lon, end_lat = result["end"]["coord"]
            start_marker_layer.location = (start_lat, start_lon)
            start_marker_layer.radius = 9
            start_marker_layer.opacity = 1.0
            start_marker_layer.fill_opacity = 1.0

            end_marker_layer.location = (end_lat, end_lon)
            end_marker_layer.radius = 9
            end_marker_layer.opacity = 1.0
            end_marker_layer.fill_opacity = 1.0

            if route_positions:
                plus15_map_widget.center = (
                    sum(pt[0] for pt in route_positions) / len(route_positions),
                    sum(pt[1] for pt in route_positions) / len(route_positions),
                )

        with ui.card(full_screen=True, height="700px"):
            ui.card_header("Plus 15")

            @render_widget
            def plus15_map():
                return plus15_map_widget

    # -------------------------------------------------------------------------
    # Existing Calgary Transit Dashboard (retained)
    # -------------------------------------------------------------------------
    with ui.nav_panel("Calgary Transit Dashboard"):
        ui.h1("Calgary Transit Dashboard")

        transit_datasets = {
            name: load_dataset(name, url) for name, url in DATASET_URLS.items()
        }
        ridership_df = transit_datasets["ridership"]
        ctrain_df = transit_datasets["ctrain"]

        CTRAIN_COORDS = {
            "1st St SW": (51.0460, -114.0656),
            "39 Avenue": (51.0183, -114.0622),
            "3rd St SW": (51.0461, -114.0694),
            "45 St SW": (51.0378, -114.1528),
            "4th St SW": (51.0461, -114.0718),
            "69 St SW": (51.0373, -114.1878),
            "6th St SW": (51.0462, -114.0754),
            "7th St SW": (51.0462, -114.0789),
            "8th St SW": (51.0463, -114.0811),
            "Anderson": (50.9567, -114.0739),
            "Banff Trail": (51.0708, -114.1200),
            "Barlow/Max Bell": (51.0456, -113.9997),
            "Brentwood": (51.0874, -114.1294),
            "Bridgeland/Memorial": (51.0485, -114.0436),
            "Canyon Meadows": (50.9328, -114.0714),
            "Centre Street": (51.0461, -114.0625),
            "Chinook": (50.9972, -114.0683),
            "City Hall": (51.0458, -114.0573),
            "Crowfoot": (51.1228, -114.2072),
            "Dalhousie": (51.1033, -114.1606),
            "Downtown West/Kerby": (51.0466, -114.0886),
            "Erlton/Stampede": (51.0336, -114.0603),
            "Fish Creek/Lacombe": (50.9239, -114.0711),
            "Franklin": (51.0481, -113.9931),
            "Heritage": (50.9786, -114.0711),
            "Lions Park": (51.0656, -114.0983),
            "Marlborough": (51.0583, -113.9806),
            "Martindale": (51.1147, -113.9619),
            "McKnight-Westwinds": (51.0978, -113.9722),
            "Rundle": (51.0753, -113.9664),
            "Saddletowne": (51.1258, -113.9483),
            "SAIT/AUARTS/Jubilee": (51.0628, -114.0908),
            "Shaganappi Point": (51.0417, -114.1189),
            "Shawnessy": (50.9064, -114.0708),
            "Sirocco": (51.0381, -114.1678),
            "Somerset-Bridlewood": (50.8989, -114.0700),
            "Southland": (50.9639, -114.0725),
            "Sunalta": (51.0456, -114.0989),
            "Sunnyside": (51.0558, -114.0850),
            "Tuscany": (51.1344, -114.2378),
            "University": (51.0792, -114.1331),
            "Victoria Park/Stampede": (51.0378, -114.0581),
            "Westbrook": (51.0422, -114.1286),
            "Whitehorn": (51.0861, -113.9622),
            "Zoo": (51.0472, -114.0256),
        }

        with ui.card(full_screen=True, height="350px"):
            ui.card_header("2025 C-Train Boardings")

            @render_widget
            def boarding_map():
                from ipyleaflet import CircleMarker as LeafletCircleMarker
                from ipyleaflet import Map as LeafletMap

                map_widget = LeafletMap(
                    center=(51.0447, -114.0719), zoom=12, scroll_wheel_zoom=True
                )
                df = ctrain_df.copy()
                if df.empty:
                    return map_widget

                station_col = "station" if "station" in df.columns else df.columns[0]
                boardings_col = (
                    "boardings" if "boardings" in df.columns else df.columns[1]
                )
                df["station_clean"] = df[station_col].astype(str).str.strip()
                df["boardings_clean"] = pd.to_numeric(
                    df[boardings_col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ).fillna(0)
                min_b = max(1, float(df["boardings_clean"].min()))
                max_b = max(min_b + 1, float(df["boardings_clean"].max()))

                for _, row in df.iterrows():
                    station = row["station_clean"]
                    boardings = int(row["boardings_clean"])
                    coords = CTRAIN_COORDS.get(station)
                    if not coords:
                        continue
                    norm = (math.sqrt(max(0, boardings)) - math.sqrt(min_b)) / (
                        math.sqrt(max_b) - math.sqrt(min_b)
                    )
                    radius = int(3 + max(0, norm) * 10)
                    color = (
                        "#2A4EB1"
                        if boardings >= 2_000_000
                        else ("#2152D8" if boardings >= 500_000 else "#0D277A")
                    )
                    marker = LeafletCircleMarker(
                        location=coords,
                        radius=radius,
                        color="#000000",
                        fill_color=color,
                        fill_opacity=0.7,
                        weight=1,
                    )
                    marker.popup = HTML(
                        value=f"<b>{station}</b><br/>Annual Boardings: {boardings:,}"
                    )
                    map_widget.add_layer(marker)
                return map_widget

        summary = ridership_df.copy()
        if not summary.empty and {"year", "ridership"}.issubset(summary.columns):
            summary["year"] = pd.to_numeric(summary["year"], errors="coerce")
            summary["ridership"] = pd.to_numeric(summary["ridership"], errors="coerce")
            summary = summary.dropna(subset=["year", "ridership"])
            annual = (
                summary.groupby("year", as_index=False)["ridership"]
                .sum()
                .sort_values("year")
            )
        else:
            annual = pd.DataFrame(columns=["year", "ridership"])

        if not annual.empty:
            min_year = int(annual["year"].min())
            max_year = int(annual["year"].max())
        else:
            min_year, max_year = 2000, 2025

        with ui.card(full_screen=True, height="350px"):
            ui.card_header("Annual Transit Ridership")
            ui.input_slider(
                "year_range",
                "Filter Years:",
                min=min_year,
                max=max_year,
                value=[min_year, max_year],
                sep="",
            )

            @render.ui
            def render_plot():
                if annual.empty:
                    return ui.p("Ridership data is currently unavailable.")

                start_year, end_year = input.year_range()
                df = annual[
                    (annual["year"] >= start_year) & (annual["year"] <= end_year)
                ].copy()
                if df.empty:
                    return ui.p("No ridership records match this year range.")

                df["year_str"] = df["year"].astype(int).astype(str)
                palette = ["#40c2f3", "#409ff3", "#407cf3", "#205dc7"]
                df["color"] = [palette[i % len(palette)] for i in range(len(df))]
                source = ColumnDataSource(df)
                plot = figure(
                    title="Annual Transit Ridership",
                    x_range=df["year_str"].tolist(),
                    x_axis_label="Year",
                    y_axis_label="Total Ridership",
                    height=350,
                    sizing_mode="stretch_width",
                    tools="",
                )
                plot.vbar(
                    x="year_str",
                    top="ridership",
                    width=0.6,
                    source=source,
                    fill_color="color",
                    line_color="white",
                    line_width=1,
                    hover_fill_color="#18418c",
                    hover_line_color="#40c2f3",
                )
                plot.add_tools(HoverTool(tooltips="Year @year: @ridership{0,0}"))
                plot.xaxis.major_label_orientation = 1
                plot.xgrid.grid_line_color = None
                plot.ygrid.grid_line_dash = "dashed"
                plot.ygrid.grid_line_alpha = 0.4
                plot.outline_line_color = None
                plot.background_fill_color = "#ffffff"
                html_content = file_html(plot, CDN, "Transit Ridership Plot")
                return ui.tags.iframe(
                    srcdoc=html_content,
                    style="width: 100%; height: 320px; border: none; overflow: hidden;",
                )

        budget_df = transit_datasets["budget"]

        # Set MIN_SLICES = 8 to keep Green Line (#1), Calgary Transit (#2),
        # and Mobility (#7) + "Other" visible (or set to 4 for Calgary Transit only).
        MIN_SLICES = 8

        with ui.card(full_screen=True, height="520px"):
            ui.card_header("Transit's Portion of the Budget")

            ui.input_slider(
                "budget_slices",
                "Number of Pie Slices:",
                min=MIN_SLICES,
                max=15,
                value=MIN_SLICES,
                step=1,
            )

            @render.ui
            def render_budget_pie():
                import json

                if budget_df.empty or "budget" not in budget_df.columns:
                    return ui.p("Budget data is currently unavailable.")

                df = budget_df.copy()
                df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0.0)
                unit_col = (
                    "business_unit" if "business_unit" in df.columns else df.columns[0]
                )

                grouped = (
                    df.groupby(unit_col, as_index=False)["budget"]
                    .sum()
                    .sort_values("budget", ascending=False)
                    .reset_index(drop=True)
                )

                # Enforce minimum of MIN_SLICES (Top N-1 units + 1 "Other" slice)
                n_slices = max(MIN_SLICES, int(input.budget_slices()))
                top_n = grouped.iloc[: n_slices - 1].copy()
                other_sum = float(grouped.iloc[n_slices - 1 :]["budget"].sum())

                blue_palette = ["#40c2f3", "#409ff3", "#407cf3", "#205dc7"]
                series_data = []
                color_idx = 0

                for _, row in top_n.iterrows():
                    name = str(row[unit_col]).strip()
                    val = round(float(row["budget"]), 2)
                    is_transit = name == "Calgary Transit"
                    is_exploded = name in {"Calgary Transit", "Green Line"}

                    if name == "Green Line":
                        slice_color = "#10B981"  # Emerald Green for Green Line
                    elif is_transit:
                        slice_color = "#00C49A"  # Mint/Teal Green for Calgary Transit
                    else:
                        slice_color = blue_palette[color_idx % len(blue_palette)]
                        color_idx += 1

                    series_data.append(
                        {
                            "name": name,
                            "y": val,
                            "color": slice_color,
                            "sliced": is_exploded,
                            "selected": is_exploded,
                        }
                    )

                if other_sum > 0:
                    remaining_count = len(grouped) - (n_slices - 1)
                    series_data.append(
                        {
                            "name": f"Other ({remaining_count} Business Units)",
                            "y": round(other_sum, 2),
                            "color": "#1E3A8A",
                            "sliced": False,
                        }
                    )

                data_json = json.dumps(series_data)

                html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://code.highcharts.com/highcharts.js"></script>
  <script src="https://code.highcharts.com/highcharts-3d.js"></script>
  <style>
    html, body, #container {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f172a;
      overflow: hidden;
      font-family: sans-serif;
    }}
  </style>
</head>
<body>
  <div id="container"></div>
  <script>
    Highcharts.chart('container', {{
      chart: {{
        type: 'pie',
        backgroundColor: '#0f172a',
        options3d: {{
          enabled: true,
          alpha: 45,
          beta: 0
        }}
      }},
      title: {{
        text: "Transit's Portion of City of Calgary Budget ($ Millions)",
        style: {{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold' }}
      }},
      subtitle: {{
        text: "Click any wedge to explode",
        style: {{ color: '#00E5FF', fontSize: '12px' }}
      }},
      tooltip: {{
        backgroundColor: '#1e293b',
        borderColor: '#00C49A',
        style: {{ color: '#f8fafc' }},
        pointFormat: '<b>${{point.y:,.2f}}M</b> ({{point.percentage:.1f}}% of total budget)'
      }},
      plotOptions: {{
        pie: {{
          allowPointSelect: true,
          cursor: 'pointer',
          depth: 45,
          slicedOffset: 22,
          edgeColor: '#0f172a',
          edgeWidth: 1.5,
          dataLabels: {{
            enabled: true,
            format: '<b>{{point.name}}</b>: ${{point.y:,.1f}}M ({{point.percentage:.1f}}%)',
            style: {{
              color: '#f8fafc',
              fontSize: '11px',
              textOutline: '1px #0f172a'
            }}
          }}
        }}
      }},
      credits: {{ enabled: false }},
      series: [{{
        name: 'Budget ($M)',
        data: {data_json}
      }}]
    }});
  </script>
</body>
</html>"""

                return ui.tags.iframe(
                    srcdoc=html_doc,
                    style="width: 100%; height: 410px; border: none; overflow: hidden; border-radius: 8px;",
                )

        with ui.card(full_screen=True, height="480px"):
            ui.card_header("Transit Ridership by Fare Category")

            @render.ui
            def render_fare_doughnut():
                import json

                if ridership_df.empty:
                    return ui.p("Ridership data is currently unavailable.")

                fare_cols = {
                    "adult_ridership": ("Adult Ridership", "#205dc7"),
                    "youth_ridership": ("Youth Ridership", "#409ff3"),
                    "u_pass_ridership": ("U-Pass Ridership", "#00E5FF"),
                    "seniors_pass_ridership": ("Seniors Pass Ridership", "#10B981"),
                }

                df = ridership_df.copy()
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
                for col in fare_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        df[col] = 0.0

                valid_df = df.dropna(subset=list(fare_cols.keys()), how="all")
                if valid_df.empty:
                    return ui.p(
                        "Fare category ridership data is currently unavailable."
                    )

                latest_year = int(valid_df["year"].max())
                df_latest = valid_df[valid_df["year"] == latest_year]

                series_data = []
                total_rides = 0.0
                for col, (label, color) in fare_cols.items():
                    annual_sum = float(df_latest[col].sum())
                    total_rides += annual_sum
                    series_data.append(
                        {
                            "name": label,
                            "y": round(annual_sum / 1_000_000.0, 1),
                            "color": color,
                        }
                    )

                # Displays "42.8 M" (or change to "4.28 M" if you prefer that exact text)
                center_total = f"{total_rides / 1_000_000.0:.1f} M"
                data_json = json.dumps(series_data)

                html_doc = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <script src="https://code.highcharts.com/highcharts-3d.js"></script>
    <style>
        html, body, #doughnut-container {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        background: #0f172a;
        overflow: hidden;
        font-family: sans-serif;
        position: relative;
        }}
        #center-kpi {{
        position: absolute;
        top: 54%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        pointer-events: none;
        z-index: 10;
        }}
    </style>
    </head>
    <body>
    <div id="center-kpi">
        <div style="color: #40c2f3; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;">
        {latest_year} TOTAL
        </div>
        <div style="color: #f8fafc; font-size: 26px; font-weight: bold; margin-top: 2px;">
        {center_total}
        </div>
    </div>
    <div id="doughnut-container"></div>
    <script>
        Highcharts.chart('doughnut-container', {{
        chart: {{
            type: 'pie',
            backgroundColor: '#0f172a',
            options3d: {{
            enabled: true,
            alpha: 45,
            beta: 0
            }}
        }},
        title: {{
            text: "Transit Ridership by Fare Category",
            style: {{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold' }}
        }},
        subtitle: {{
            text: null
        }},
        tooltip: {{
            backgroundColor: '#1e293b',
            borderColor: '#00E5FF',
            style: {{ color: '#f8fafc' }},
            pointFormat: '<b>{{point.y:.1f}} million</b> ({{point.percentage:.1f}}%)'
        }},
        plotOptions: {{
            pie: {{
            innerSize: '62%',
            depth: 45,
            allowPointSelect: true,
            cursor: 'pointer',
            edgeColor: '#0f172a',
            edgeWidth: 1.5,
            showInLegend: false,
            dataLabels: {{
                enabled: true,
                format: '<b>{{point.name}}</b>: {{point.y:.1f}} million ({{point.percentage:.1f}}%)',
                style: {{
                color: '#f8fafc',
                fontSize: '11px',
                textOutline: '1px #0f172a'
                }}
            }}
            }}
        }},
        credits: {{ enabled: false }},
        series: [{{
            name: 'Ridership',
            data: {data_json}
        }}]
        }});
    </script>
    </body>
    </html>"""

                return ui.tags.iframe(
                    srcdoc=html_doc,
                    style="width: 100%; height: 410px; border: none; overflow: hidden; border-radius: 8px;",
                )

        # Prepare monthly date bounds for the 10-year / 12-month slider
        monthly_df = ridership_df.copy()
        if not monthly_df.empty and {
            "date",
            "ctrain_boarding_passengers",
            "bus_boarding_passengers",
        }.issubset(monthly_df.columns):
            monthly_df["ctrain_boarding_passengers"] = pd.to_numeric(
                monthly_df["ctrain_boarding_passengers"], errors="coerce"
            )
            monthly_df["bus_boarding_passengers"] = pd.to_numeric(
                monthly_df["bus_boarding_passengers"], errors="coerce"
            )
            monthly_df["date_str"] = monthly_df["date"].astype(str).str.strip()
            monthly_df["dt"] = pd.to_datetime(
                monthly_df["date_str"], format="%Y/%m", errors="coerce"
            )
            monthly_df = (
                monthly_df.dropna(subset=["dt"])
                .dropna(
                    subset=["ctrain_boarding_passengers", "bus_boarding_passengers"],
                    how="all",
                )
                .sort_values("dt")
                .reset_index(drop=True)
            )
        else:
            monthly_df = pd.DataFrame(
                columns=[
                    "date_str",
                    "dt",
                    "ctrain_boarding_passengers",
                    "bus_boarding_passengers",
                ]
            )

        if not monthly_df.empty:
            # Restrict slider to the last 10 years (120 months) and default to the last 12 months
            monthly_10y = monthly_df.tail(120).reset_index(drop=True)
            max_month_date = monthly_10y["dt"].iloc[-1].date()
            min_month_date = monthly_10y["dt"].iloc[0].date()
            default_start_date = (
                monthly_10y["dt"].iloc[-12].date()
                if len(monthly_10y) >= 12
                else min_month_date
            )
        else:
            min_month_date = date(2015, 7, 1)
            max_month_date = date(2025, 6, 1)
            default_start_date = date(2024, 7, 1)

        with ui.card(full_screen=True, height="520px"):
            ui.card_header("C-Train and Bus Ridership")

            ui.input_slider(
                "monthly_range",
                "Filter Months:",
                min=min_month_date,
                max=max_month_date,
                value=[default_start_date, max_month_date],
                time_format="%Y/%m",
            )

            @render.ui
            def render_ctrain_bus_line():
                import json

                if monthly_df.empty:
                    return ui.p(
                        "Monthly C-Train and Bus ridership data is unavailable."
                    )

                start_dt, end_dt = input.monthly_range()
                start_ym = start_dt.strftime("%Y/%m")
                end_ym = end_dt.strftime("%Y/%m")

                df_sub = monthly_df[
                    (monthly_df["date_str"] >= start_ym)
                    & (monthly_df["date_str"] <= end_ym)
                ].copy()

                if df_sub.empty:
                    return ui.p("No monthly records match this date range.")

                # Show vertex dots on shorter views (e.g., 1-2 years) for crisp angles
                show_dots = "false"

                dates_json = json.dumps(df_sub["date_str"].tolist())
                ctrain_json = json.dumps(
                    [
                        round(float(v), 0) if pd.notna(v) else None
                        for v in df_sub["ctrain_boarding_passengers"]
                    ]
                )
                bus_json = json.dumps(
                    [
                        round(float(v), 0) if pd.notna(v) else None
                        for v in df_sub["bus_boarding_passengers"]
                    ]
                )

                html_doc = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        html, body, #line-container {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        background: #0f172a;
        overflow: hidden;
        font-family: sans-serif;
        }}
    </style>
    </head>
    <body>
    <div id="line-container"></div>
    <script>
        const chartDom = document.getElementById('line-container');
        const myChart = echarts.init(chartDom);

        const option = {{
        backgroundColor: '#0f172a',
        title: {{
            text: 'C-Train and Bus Ridership',
            left: 'center',
            top: 10,
            textStyle: {{
            color: '#f8fafc',
            fontSize: 16,
            fontWeight: 'bold'
            }}
        }},
        tooltip: {{
            trigger: 'axis',
            backgroundColor: '#1e293b',
            borderColor: '#10B981',
            borderWidth: 1.5,
            textStyle: {{ color: '#f8fafc', fontSize: 12 }}
        }},
        legend: {{
            top: 38,
            left: 'center',
            textStyle: {{ color: '#f8fafc', fontSize: 12 }},
            data: ['C-Train Ridership', 'Bus Ridership']
        }},
        grid: {{
            left: '6%',
            right: '4%',
            top: 78,
            bottom: 40,
            containLabel: true
        }},
        xAxis: {{
            type: 'category',
            boundaryGap: ['3%', '3%'],
            data: {dates_json},
            splitLine: {{ show: false }},
            axisLine: {{ lineStyle: {{ color: '#334155' }} }},
            axisLabel: {{ color: '#94a3b8', fontSize: 11 }}
        }},
        yAxis: {{
            type: 'value',
            scale: true,
            splitLine: {{ show: false }},
            axisLine: {{ show: true, lineStyle: {{ color: '#334155' }} }},
            axisLabel: {{
            color: '#94a3b8',
            fontSize: 11,
            formatter: function(val) {{
                return (val / 1000000).toFixed(1) + 'M';
            }}
            }}
        }},
        series: [
            {{
            name: 'C-Train Ridership',
            type: 'line',
            smooth: false,
            symbol: 'circle',
            showSymbol: {show_dots},
            symbolSize: 7,
            itemStyle: {{ color: '#10B981' }},
            lineStyle: {{ width: 3, color: '#10B981', join: 'miter' }},
            data: {ctrain_json}
            }},
            {{
            name: 'Bus Ridership',
            type: 'line',
            smooth: false,
            symbol: 'circle',
            showSymbol: {show_dots},
            symbolSize: 7,
            itemStyle: {{ color: '#00E5FF' }},
            lineStyle: {{ width: 3, color: '#00E5FF', join: 'miter' }},
            data: {bus_json}
            }}
        ]
        }};

        myChart.setOption(option);
        window.addEventListener('resize', () => myChart.resize());
    </script>
    </body>
    </html>"""

                return ui.tags.iframe(
                    srcdoc=html_doc,
                    style="width: 100%; height: 400px; border: none; overflow: hidden; border-radius: 8px;",
                )

        # Prepare complete calendar years (2014 through 2024, excluding incomplete 2025)
        ys_df = ridership_df.copy()
        ys_cols = [
            "youth_monthly_pass_ridership",
            "youth_low_income_monthly_pass_ridership",
            "seniors_regular_pass_ridership",
            "seniors_low_income_pass_ridership",
        ]
        if not ys_df.empty and {"year", *ys_cols}.issubset(ys_df.columns):
            ys_df["year"] = pd.to_numeric(ys_df["year"], errors="coerce")
            for col in ys_cols:
                ys_df[col] = pd.to_numeric(ys_df[col], errors="coerce").fillna(0.0)
            ys_df = ys_df[(ys_df["year"] >= 2014) & (ys_df["year"] <= 2024)].copy()
            ys_df["year"] = ys_df["year"].astype(int)
            ys_annual = (
                ys_df.groupby("year", as_index=False)[ys_cols]
                .sum()
                .sort_values("year")
                .reset_index(drop=True)
            )
        else:
            ys_annual = pd.DataFrame(columns=["year", *ys_cols])

        if not ys_annual.empty:
            min_ys_year = int(ys_annual["year"].min())
            max_ys_year = int(ys_annual["year"].max())
            default_start_ys_year = max(
                min_ys_year, max_ys_year - 6
            )  # 2018–2024 (7 years)
        else:
            min_ys_year, max_ys_year, default_start_ys_year = 2014, 2024, 2018

        with ui.card(full_screen=True, height="540px"):
            ui.card_header("Youth vs Senior Ridership")

            ui.input_slider(
                "youth_senior_years",
                "Filter Years:",
                min=min_ys_year,
                max=max_ys_year,
                value=[default_start_ys_year, max_ys_year],
                sep="",
                step=1,
            )

            @render.ui
            def render_youth_senior_chart():
                import json

                if ys_annual.empty:
                    return ui.p("Youth and Senior ridership data is unavailable.")

                start_yr, end_yr = input.youth_senior_years()
                # Enforce a minimum span of 3 years
                if end_yr - start_yr < 2:
                    if end_yr - 2 >= min_ys_year:
                        start_yr = end_yr - 2
                    else:
                        end_yr = min(max_ys_year, start_yr + 2)

                df_sub = ys_annual[
                    (ys_annual["year"] >= start_yr) & (ys_annual["year"] <= end_yr)
                ].reset_index(drop=True)

                if df_sub.empty:
                    return ui.p("No records match this year range.")

                years_json = json.dumps(df_sub["year"].astype(str).tolist())
                youth_reg_json = json.dumps(
                    [int(round(v)) for v in df_sub["youth_monthly_pass_ridership"]]
                )
                youth_low_json = json.dumps(
                    [
                        int(round(v))
                        for v in df_sub["youth_low_income_monthly_pass_ridership"]
                    ]
                )
                senior_reg_json = json.dumps(
                    [int(round(v)) for v in df_sub["seniors_regular_pass_ridership"]]
                )
                senior_low_json = json.dumps(
                    [int(round(v)) for v in df_sub["seniors_low_income_pass_ridership"]]
                )

                html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://code.highcharts.com/highcharts.js"></script>
  <style>
    html, body, #ys-container {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f172a;
      overflow: hidden;
      font-family: sans-serif;
    }}
  </style>
</head>
<body>
  <div id="ys-container"></div>
  <script>
    Highcharts.chart('ys-container', {{
      chart: {{
        type: 'column',
        backgroundColor: '#0f172a'
      }},
      title: {{
        text: 'Youth vs Senior Ridership',
        style: {{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold' }}
      }},
      xAxis: {{
        categories: {years_json},
        title: {{
          text: 'Year',
          style: {{ color: '#cbd5e1', fontSize: '12px', fontWeight: 'bold' }}
        }},
        lineColor: '#334155',
        tickColor: '#334155',
        labels: {{
          style: {{ color: '#f8fafc', fontSize: '11px', fontWeight: 'bold' }}
        }}
      }},
      yAxis: {{
        min: 0,
        gridLineWidth: 0,
        lineWidth: 1,
        lineColor: '#334155',
        title: {{
          text: 'Annual Ridership',
          style: {{ color: '#cbd5e1', fontSize: '12px', fontWeight: 'bold' }}
        }},
        labels: {{
          style: {{ color: '#94a3b8', fontSize: '11px' }},
          formatter: function() {{
            return (this.value / 1000000).toFixed(1) + 'M';
          }}
        }}
      }},
      legend: {{
        enabled: true,
        itemStyle: {{ color: '#f8fafc', fontSize: '11px', fontWeight: '600' }},
        itemHoverStyle: {{ color: '#00E5FF' }}
      }},
      tooltip: {{
        backgroundColor: '#1e293b',
        borderColor: '#00E5FF',
        borderWidth: 1.5,
        style: {{ color: '#f8fafc', fontSize: '12px' }},
        formatter: function() {{
          const parts = this.series.name.split(' — ');
          const passengerCategory = parts[0] || this.series.userOptions.stack;
          const fareType = parts[1] || this.series.name;
          return '<b>Year:</b> ' + this.x + '<br/>' +
                 '<b>Passenger Category:</b> ' + passengerCategory + '<br/>' +
                 '<b>Fare Type:</b> ' + fareType + '<br/>' +
                 '<b>Annual Ridership:</b> ' + Highcharts.numberFormat(this.y, 0, '.', ',') +
                 ' (' + (this.y / 1000000).toFixed(2) + 'M)<br/>' +
                 '<b>' + passengerCategory + ' Group Total:</b> ' + Highcharts.numberFormat(this.point.stackTotal, 0, '.', ',');
        }}
      }},
      plotOptions: {{
        column: {{
          stacking: 'normal',
          borderColor: '#0f172a',
          borderWidth: 1.5,
          borderRadius: 4,
          groupPadding: 0.16,
          pointPadding: 0.06
        }}
      }},
      credits: {{ enabled: false }},
      series: [
        {{
          name: 'Youth — Regular',
          data: {youth_reg_json},
          stack: 'Youth',
          color: '#00E5FF',
          legendIndex: 1
        }},
        {{
          name: 'Youth — Low Income',
          data: {youth_low_json},
          stack: 'Youth',
          color: '#0284C7',
          legendIndex: 0
        }},
        {{
          name: 'Senior — Regular',
          data: {senior_reg_json},
          stack: 'Senior',
          color: '#10B981',
          legendIndex: 3
        }},
        {{
          name: 'Senior — Low Income',
          data: {senior_low_json},
          stack: 'Senior',
          color: '#047857',
          legendIndex: 2
        }}
      ]
    }});
  </script>
</body>
</html>"""

                return ui.tags.iframe(
                    srcdoc=html_doc,
                    style="width: 100%; height: 420px; border: none; overflow: hidden; border-radius: 8px;",
                )

        with ui.card(full_screen=True, height="560px"):
            ui.card_header("Monthly Ridership Seasonality")

            @render.ui
            def render_polar_seasonality():
                import json

                if ridership_df.empty or not {"year", "month", "ridership"}.issubset(
                    ridership_df.columns
                ):
                    return ui.p("Ridership data is currently unavailable.")

                df = ridership_df.copy()
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
                df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce")

                # Filter to complete calendar years 2007 through 2024
                df = df[(df["year"] >= 2007) & (df["year"] <= 2024)].dropna(
                    subset=["year", "month", "ridership"]
                )
                if df.empty:
                    return ui.p("No ridership records found for 2007–2024.")

                month_order = [
                    ("Jan", "January"),
                    ("Feb", "February"),
                    ("Mar", "March"),
                    ("Apr", "April"),
                    ("May", "May"),
                    ("Jun", "June"),
                    ("Jul", "July"),
                    ("Aug", "August"),
                    ("Sep", "September"),
                    ("Oct", "October"),
                    ("Nov", "November"),
                    ("Dec", "December"),
                ]

                df["month_abbr"] = (
                    df["month"].astype(str).str.strip().str[:3].str.title()
                )

                categories = []
                column_data = []
                line_data = []

                for abbr, full_name in month_order:
                    m_vals = df.loc[df["month_abbr"] == abbr, "ridership"]
                    avg_val = float(m_vals.mean()) if not m_vals.empty else 0.0
                    avg_rounded = int(round(avg_val))

                    # Assign dashboard palette colors by seasonal ridership tier
                    if avg_val >= 8_200_000:
                        color = "#10B981"  # Emerald Green (Peak Fall: Sep, Oct, Nov)
                    elif avg_val >= 7_800_000:
                        color = (
                            "#00E5FF"  # Bright Turquoise (Winter/Spring Peak: Feb, Mar)
                        )
                    elif avg_val >= 7_350_000:
                        color = "#40c2f3"  # Sky Blue (Shoulder: Jan, Apr, Dec)
                    elif avg_val >= 7_000_000:
                        color = "#407cf3"  # Medium Blue (Early Summer: May, Jun, Jul)
                    else:
                        color = "#205dc7"  # Deep Cobalt Blue (August Trough)

                    categories.append(full_name)
                    column_data.append(
                        {"name": full_name, "y": avg_rounded, "color": color}
                    )
                    line_data.append(avg_rounded)

                overall_avg = (
                    int(round(sum(line_data) / len(line_data))) if line_data else 0
                )
                categories_json = json.dumps(categories)
                column_json = json.dumps(column_data)
                line_json = json.dumps(line_data)

                html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://code.highcharts.com/highcharts.js"></script>
  <script src="https://code.highcharts.com/highcharts-more.js"></script>
  <style>
    html, body, #polar-container {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f172a;
      overflow: hidden;
      font-family: sans-serif;
    }}
  </style>
</head>
<body>
  <div id="polar-container"></div>
  <script>
    Highcharts.chart('polar-container', {{
      chart: {{
        polar: true,
        backgroundColor: '#0f172a',
        marginTop: 70,
        marginBottom: 45
      }},
      title: {{
        text: 'Monthly Ridership Seasonality (2007–2024)',
        style: {{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold' }}
      }},
      pane: {{
        size: '82%',
        innerSize: '22%',
        startAngle: 0,
        endAngle: 360
      }},
      xAxis: {{
        categories: {categories_json},
        tickmarkPlacement: 'on',
        lineWidth: 1,
        lineColor: '#334155',
        gridLineColor: '#1e293b',
        labels: {{
          style: {{ color: '#f8fafc', fontSize: '11px', fontWeight: 'bold' }},
          distance: 18
        }}
      }},
      yAxis: {{
        min: 4500000,
        max: 9000000,
        tickInterval: 1500000,
        gridLineInterpolation: 'circle',
        gridLineColor: '#1e293b',
        gridLineDashStyle: 'Dash',
        labels: {{
          style: {{ color: '#94a3b8', fontSize: '10px' }},
          formatter: function() {{
            return (this.value / 1000000).toFixed(1) + 'M';
          }}
        }},
        plotLines: [{{
          value: {overall_avg},
          color: '#f8fafc',
          dashStyle: 'ShortDash',
          width: 1.5,
          zIndex: 5
        }}]
      }},
      tooltip: {{
        shared: true,
        backgroundColor: '#1e293b',
        borderColor: '#00E5FF',
        borderWidth: 1.5,
        style: {{ color: '#f8fafc', fontSize: '12px' }},
        formatter: function() {{
          const pt = this.points[0];
          const millions = (pt.y / 1000000).toFixed(2) + 'M';
          const diffPct = (((pt.y - {overall_avg}) / {overall_avg}) * 100).toFixed(1);
          const sign = diffPct >= 0 ? '+' : '';
          return '<b>' + this.x + ' (2007–2024 Avg)</b><br/>' +
                 '<b>Average Ridership:</b> ' + Highcharts.numberFormat(pt.y, 0, '.', ',') + ' (' + millions + ')<br/>' +
                 '<b>vs. 18-Year Monthly Mean:</b> ' + sign + diffPct + '%';
        }}
      }},
      legend: {{
        enabled: true,
        itemStyle: {{ color: '#f8fafc', fontSize: '11px', fontWeight: '600' }},
        itemHoverStyle: {{ color: '#00E5FF' }}
      }},
      plotOptions: {{
        column: {{
          pointPadding: 0.04,
          groupPadding: 0.02,
          borderWidth: 1.5,
          borderColor: '#0f172a',
          borderRadius: 4
        }},
        series: {{
          animation: true
        }}
      }},
      credits: {{ enabled: false }},
      series: [
        {{
          type: 'column',
          name: 'Monthly Average Ridership',
          color: '#10B981',
          data: {column_json},
          dataLabels: {{
            enabled: true,
            formatter: function() {{
              return (this.y / 1000000).toFixed(2) + 'M';
            }},
            style: {{
              color: '#f8fafc',
              fontSize: '10px',
              fontWeight: 'bold',
              textOutline: '1px #0f172a'
            }}
          }}
        }},
        {{
          type: 'line',
          name: 'Seasonal Trend Contour',
          data: {line_json},
          color: '#00E5FF',
          lineWidth: 2.5,
          marker: {{
            enabled: true,
            radius: 4,
            fillColor: '#f8fafc',
            lineWidth: 2,
            lineColor: '#00E5FF'
          }}
        }}
      ]
    }});
  </script>
</body>
</html>"""

                return ui.tags.iframe(
                    srcdoc=html_doc,
                    style="width: 100%; height: 480px; border: none; overflow: hidden; border-radius: 8px;",
                )

        # Prepare 10 most recent complete calendar years (2015–2024) for the Bump Chart
        bump_df = ridership_df.copy()
        bump_fare_cols = {
            "adult_low_income_monthly_pass_ridership": (
                "Adult Low Income Pass",
                "#10B981",
            ),
            "adult_monthly_pass_ridership": ("Adult Monthly Pass", "#00E5FF"),
            "adult_cash_single_ticket_ride_ridership": (
                "Adult Cash / Single Ticket",
                "#40c2f3",
            ),
            "u_pass_ridership": ("U-Pass", "#407cf3"),
            "youth_monthly_pass_ridership": ("Youth Monthly Pass", "#205dc7"),
        }

        if not bump_df.empty and {"year", *bump_fare_cols.keys()}.issubset(
            bump_df.columns
        ):
            bump_df["year"] = pd.to_numeric(bump_df["year"], errors="coerce")
            for col in bump_fare_cols:
                bump_df[col] = pd.to_numeric(bump_df[col], errors="coerce").fillna(0.0)
            bump_df = bump_df[
                (bump_df["year"] >= 2015) & (bump_df["year"] <= 2024)
            ].copy()
            bump_df["year"] = bump_df["year"].astype(int)
            bump_annual = (
                bump_df.groupby("year", as_index=False)[list(bump_fare_cols.keys())]
                .sum()
                .sort_values("year")
                .tail(10)
                .reset_index(drop=True)
            )
        else:
            bump_annual = pd.DataFrame(columns=["year", *bump_fare_cols.keys()])

        if not bump_annual.empty:
            min_bump_year = int(bump_annual["year"].min())
            max_bump_year = int(bump_annual["year"].max())
            default_start_bump_year = 2021
        else:
            min_bump_year, max_bump_year, default_start_bump_year = 2015, 2024, 2015

        with ui.card(full_screen=True, height="560px"):
            ui.card_header("Fare Type Popularity")

            ui.input_slider(
                "bump_years",
                "Filter Years:",
                min=min_bump_year,
                max=max_bump_year,
                value=[default_start_bump_year, max_bump_year],
                sep="",
                step=1,
            )

            @render.ui
            def render_fare_bump_chart():
                import json

                if bump_annual.empty:
                    return ui.p("Fare type ranking data is currently unavailable.")

                start_yr, end_yr = input.bump_years()
                # Enforce a minimum of 2 years and a maximum of 10 years
                if end_yr - start_yr < 1:
                    if end_yr - 1 >= min_bump_year:
                        start_yr = end_yr - 1
                    else:
                        end_yr = min(max_bump_year, start_yr + 1)

                df_sub = bump_annual[
                    (bump_annual["year"] >= start_yr) & (bump_annual["year"] <= end_yr)
                ].reset_index(drop=True)

                if df_sub.empty:
                    return ui.p("No records match this year range.")

                cols_list = list(bump_fare_cols.keys())
                ranks_df = (
                    df_sub[cols_list]
                    .rank(axis=1, ascending=False, method="first")
                    .astype(int)
                )

                years_json = json.dumps(df_sub["year"].astype(str).tolist())
                last_idx = len(df_sub) - 1

                series_list = []
                for col, (label, color) in bump_fare_cols.items():
                    points = []
                    for i in range(len(df_sub)):
                        rk = int(ranks_df.loc[i, col])
                        rides = int(round(float(df_sub.loc[i, col])))
                        pt = {
                            "y": rk,
                            "rides": rides,
                            "millions": f"{rides / 1_000_000:.2f}M",
                        }
                        if i == last_idx:
                            pt["dataLabels"] = {
                                "enabled": True,
                                "useHTML": True,
                                "align": "left",
                                "verticalAlign": "middle",
                                "x": 16,
                                "y": 0,
                                "overflow": "allow",
                                "crop": False,
                                "formatter": None,
                                "format": (
                                    f"<div style='background:#1e293b;color:#f8fafc;"
                                    f"border:2px solid {color};border-radius:6px;"
                                    f"padding:4px 10px;font-size:11px;font-weight:700;"
                                    f"white-space:nowrap;letter-spacing:0.4px;'>"
                                    f"{rk}. {label.upper()}</div>"
                                ),
                            }
                        points.append(pt)

                    series_list.append(
                        {
                            "name": label,
                            "color": color,
                            "data": points,
                        }
                    )

                series_json = json.dumps(series_list)

                html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://code.highcharts.com/highcharts.js"></script>
  <style>
    html, body, #bump-container {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f172a;
      overflow: hidden;
      font-family: sans-serif;
    }}
  </style>
</head>
<body>
  <div id="bump-container"></div>
  <script>
    Highcharts.chart('bump-container', {{
      chart: {{
        type: 'spline',
        backgroundColor: '#0f172a',
        marginLeft: 65,
        marginRight: 235,
        marginTop: 75,
        marginBottom: 75
      }},
      title: {{
        text: 'Fare Type Popularity',
        style: {{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold' }}
      }},
      subtitle: {{
        text: null
      }},
      xAxis: {{
        categories: {years_json},
        opposite: true,
        lineWidth: 0,
        tickWidth: 0,
        gridLineWidth: 0,
        labels: {{
          style: {{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}
        }}
      }},
      yAxis: {{
        reversed: true,
        min: 1,
        max: 5,
        tickInterval: 1,
        startOnTick: true,
        endOnTick: true,
        gridLineWidth: 0,
        title: {{ text: null }},
        labels: {{
          useHTML: true,
          x: -12,
          formatter: function() {{
            return '<div style="background:#1e293b;color:#f8fafc;border:2px solid #40c2f3;' +
                   'border-radius:6px;padding:3px 9px;font-weight:700;font-size:12px;">' +
                   this.value + '</div>';
          }}
        }}
      }},
      legend: {{
        enabled: true,
        itemStyle: {{ color: '#f8fafc', fontSize: '11px', fontWeight: '600' }},
        itemHoverStyle: {{ color: '#00E5FF' }}
      }},
      tooltip: {{
        backgroundColor: '#1e293b',
        borderColor: '#00E5FF',
        borderWidth: 1.5,
        style: {{ color: '#f8fafc', fontSize: '12px' }},
        formatter: function() {{
          const suffixes = {{ 1: 'st', 2: 'nd', 3: 'rd', 4: 'th', 5: 'th' }};
          const ord = this.y + (suffixes[this.y] || 'th');
          return '<b>Year:</b> ' + this.x + '<br/>' +
                 '<b>Fare Type:</b> ' + this.series.name + '<br/>' +
                 '<b>Popularity Rank:</b> ' + ord + ' most used<br/>' +
                 '<b>Annual Ridership:</b> ' + Highcharts.numberFormat(this.point.rides, 0, '.', ',') +
                 ' (' + this.point.millions + ')';
        }}
      }},
      plotOptions: {{
        spline: {{
          lineWidth: 5,
          marker: {{
            enabled: true,
            symbol: 'circle',
            radius: 8,
            lineWidth: 2,
            lineColor: '#0f172a'
          }},
          states: {{
            hover: {{
              lineWidthPlus: 2
            }}
          }}
        }}
      }},
      credits: {{ enabled: false }},
      series: {series_json}
    }});
  </script>
</body>
</html>"""

                return ui.tags.iframe(
                    srcdoc=html_doc,
                    style="width: 100%; height: 430px; border: none; overflow: hidden; border-radius: 8px;",
                )

    ui.nav_spacer()
    with ui.nav_control():
        ui.input_dark_mode(id="mode", mode="dark")
