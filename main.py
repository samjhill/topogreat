import os
import json
import numpy as np
from stl import mesh
from pyproj import Transformer
import requests
import trimesh
from shapely.geometry import Point, Polygon
import pickle

import time
from tqdm import tqdm

# Helper: Save and Load Functions
def save_to_file(data, filename):
    """Save data to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_file(filename):
    """Load data from a file if it exists."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


# Step 1: Fetch DEM Data
def fetch_dem_data(bbox, resolution=100, cache_file="dem_data.pkl"):
    """
    Fetch DEM data from OpenTopoData or other APIs with a progress bar.
    """
    # Try loading from cache
    dem_data = load_from_file(cache_file)
    if dem_data is not None:
        print(f"Loaded DEM data from {cache_file}.")
        return dem_data

    # Otherwise, fetch from API
    print("Fetching DEM data...")
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.linspace(min_lat, max_lat, resolution)
    lons = np.linspace(min_lon, max_lon, resolution)

    elevations = []
    total_requests = len(lats) * len(lons)  # Total number of API calls
    progress = tqdm(total=total_requests, desc="Downloading DEM data")

    for lat in lats:
        for lon in lons:
            time.sleep(1)
            response = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}")
            if response.status_code == 200:
                elevation = response.json()['results'][0]['elevation']
                elevations.append(elevation)
            else:
                print(f"Error fetching DEM data: {lat}, {lon} {response.status_code}")
                elevations.append(0)  # Default to sea level if API fails.

            progress.update(1)  # Update the progress bar

    progress.close()  # Close the progress bar after completion

    dem_data = np.array(elevations).reshape((resolution, resolution))
    save_to_file(dem_data, cache_file)
    print(f"Saved DEM data to {cache_file}.")
    return dem_data


# Step 2: Create STL from Elevation Data
def create_terrain_stl(elevation_data, scale=1.0, cache_file="terrain.stl"):
    """
    Convert elevation data into a 3D terrain mesh.
    """
    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loaded terrain mesh from {cache_file}.")
        return trimesh.load(cache_file)

    print("Generating terrain mesh...")
    rows, cols = elevation_data.shape
    vertices = []
    faces = []

    for y in range(rows - 1):
        for x in range(cols - 1):
            z = elevation_data[y, x] * scale
            vertices.extend([
                [x, y, z],
                [x + 1, y, elevation_data[y, x + 1] * scale],
                [x, y + 1, elevation_data[y + 1, x] * scale],
                [x + 1, y + 1, elevation_data[y + 1, x + 1] * scale],
            ])
            faces.append([len(vertices) - 4, len(vertices) - 3, len(vertices) - 2])
            faces.append([len(vertices) - 2, len(vertices) - 3, len(vertices) - 1])

    terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    terrain_mesh.export(cache_file)
    print(f"Saved terrain mesh to {cache_file}.")
    return terrain_mesh


# Step 3: Fetch Points of Interest (Labels)
def fetch_pois(bbox, query="landmark", cache_file="pois.json"):
    """
    Fetch points of interest (POIs) from OpenStreetMap within a bounding box.
    """
    # Try loading from cache
    pois = load_from_file(cache_file)
    if pois is not None:
        print(f"Loaded POIs from {cache_file}.")
        return pois

    print("Fetching POIs...")
    min_lon, min_lat, max_lon, max_lat = bbox
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'format': 'json',
        'q': query,
        'bounded': 1,
        'viewbox': f"{min_lon},{max_lat},{max_lon},{min_lat}",
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        pois = [{'name': poi['display_name'], 'lat': float(poi['lat']), 'lon': float(poi['lon'])} for poi in response.json()]
        save_to_file(pois, cache_file)
        print(f"Saved POIs to {cache_file}.")
        return pois
    return []


# Step 4: Add Labels to STL
def add_labels_to_mesh(terrain_mesh, pois, scale=1.0, cache_file="labeled_terrain.stl"):
    """
    Add 3D labels to the terrain mesh based on POI locations.
    """
    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loaded labeled mesh from {cache_file}.")
        return trimesh.load(cache_file)

    print("Adding labels to terrain...")
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")  # WGS84 to Web Mercator
    label_meshes = []

    for poi in pois:
        x, y = transformer.transform(poi['lon'], poi['lat'])
        label = trimesh.creation.text(poi['name'], height=2, depth=1)
        label.apply_translation([x, y, terrain_mesh.bounds[2] + 5])
        label_meshes.append(label)

    combined_mesh = trimesh.util.concatenate([terrain_mesh] + label_meshes)
    combined_mesh.export(cache_file)
    print(f"Saved labeled mesh to {cache_file}.")
    return combined_mesh


# Main Workflow
if __name__ == "__main__":
    bbox = (-74.5, 40.5, -73.5, 41.5)  # Example: New York area

    # Step 1: Get DEM data
    dem_data = fetch_dem_data(bbox, resolution=10)

    # Step 2: Generate terrain STL
    terrain = create_terrain_stl(dem_data, scale=10.0)

    # Step 3: Get POIs
    pois = fetch_pois(bbox, query="landmark")

    # Step 4: Add labels and save final STL
    labeled_mesh = add_labels_to_mesh(terrain, pois)
