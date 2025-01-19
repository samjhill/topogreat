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
def fetch_dem_data(bbox, resolution=100, cache_file="app_cache/dem_data.pkl"):
    """
    Fetch DEM data from OpenTopoData or other APIs with a progress bar.
    """
    # Try loading from cache
    dem_data = load_from_file(cache_file)
    if dem_data is not None:
        print(f"Loaded DEM data from {cache_file}.")
        # with open(cache_file, 'rb') as f:
        #     data = pickle.load(f)
        #     import pdb; pdb.set_trace()
        return dem_data

    # Otherwise, fetch from API
    print("Fetching DEM data...")
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.linspace(min_lat, max_lat, resolution)
    lons = np.linspace(min_lon, max_lon, resolution)

    elevations = []
    total_requests = len(lats) * len(lons)
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
def create_terrain_mesh(elevation_data, z_scale=1, cache_file="app_cache/terrain.stl"):
    # # Try loading from cache
    # stl_data = load_from_file(cache_file)
    # if stl_data is not None:
    #     print(f"Loaded STL data from {cache_file}.")
    #     return trimesh.load_mesh(stl_data)
    
    rows, cols = elevation_data.shape
    elevation_data *= z_scale  # Apply the z-scale factor to the elevation data

    vertices = []
    faces = []

    # Create vertices (x, y, z positions)
    for i in range(rows):
        for j in range(cols):
            vertices.append([j, i, elevation_data[i, j]])  # Note x = j, y = i, z = elevation

    # Create faces (triangles)
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Define the corners of the quad (each grid cell)
            p1 = i * cols + j
            p2 = i * cols + (j + 1)
            p3 = (i + 1) * cols + j
            p4 = (i + 1) * cols + (j + 1)

            # Split the quad into two triangles (for a proper triangulation)
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    # Convert to numpy arrays for faces and vertices
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create mesh and assign vertices to faces
    terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[face[j]]

    # Save the generated mesh to a file
    terrain_mesh.save(cache_file)

    return terrain_mesh


# Step 3: Fetch Points of Interest (Labels)
def fetch_pois(bbox, query="landmark", cache_file="app_cache/pois.json"):
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
    headers = {
        "User-Agent": "topogreat/1.0 (samuhill@gmail.com)"
    }

    response = requests.get(url, params=params, headers=headers)
    print("pois url: ", response.url)
    if response.status_code == 200:
        pois = [{'name': poi['display_name'], 'lat': float(poi['lat']), 'lon': float(poi['lon'])} for poi in response.json()]
        save_to_file(pois, cache_file)
        print(f"Saved POIs to {cache_file}.")
        return pois
    else:
        print(f"error fetching POIs")
        print(response.status_code)
        import pdb; pdb.set_trace()

    return []


# Function to create 3D text mesh
def create_3d_text(origin, text, font_size=1.0, depth=2):
    # Create a Text entity
    text_entity = trimesh.path.entities.Text(origin=origin, height=5, text=text)
    
    # Convert the 2D text path into polygons (2D)
    import pdb; pdb.set_trace()
    path = text_entity.entities[0]  # Get the first path
    path_polygons = path.polygons  # Extract the polygons from the path
    
    # Now we need to extrude the 2D polygons into 3D by adding depth
    # Create a mesh for each polygon (extrude it along the z-axis)
    meshes = []
    for polygon in path_polygons:
        # Create a mesh from the polygon and extrude it
        polygon_mesh = trimesh.creation.extrude_polygon(polygon, height=depth)
        meshes.append(polygon_mesh)

    # Combine all extruded meshes into a single mesh
    text_mesh = trimesh.util.concatenate(meshes)
    
    return text_mesh

# Step 4: Add Labels to STL
def add_labels_to_mesh(terrain_mesh, pois, scale=1.0, cache_file="app_cache/labeled_terrain.stl"):
    """
    Add 3D labels to the terrain mesh based on POI locations.
    """
    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loaded labeled mesh from {cache_file}.")
        return trimesh.load(cache_file)

    print("Adding labels to terrain...")
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")  # WGS84 to Web Mercator

    for poi in pois:
        x, y = transformer.transform(poi['lon'], poi['lat'])
        label_mesh = create_3d_text(origin=x, text=poi['name'])
        terrain_mesh = trimesh.util.concatenate(terrain_mesh, label_mesh)

    terrain_mesh.export(cache_file)
    # import pdb; pdb.set_trace()
    # combined_mesh.export(cache_file)
    print(f"Saved labeled mesh to {cache_file}.")
    return terrain_mesh


# Main Workflow
if __name__ == "__main__":
    bbox = (-74.5, 40.5, -73.0, 41.5)

    # Step 1: Get DEM data
    dem_data = fetch_dem_data(bbox, resolution=10)

    # Step 2: Generate terrain STL
    terrain = create_terrain_mesh(dem_data, z_scale=.01)

    # Step 3: Get POIs
    pois = fetch_pois(bbox, query="landmark")

    # # Step 4: Add labels and save final STL
    labeled_mesh = add_labels_to_mesh(terrain, pois)
