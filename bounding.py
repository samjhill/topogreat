from geopy.geocoders import Nominatim

# Function to geocode an address and create a bounding box around it
def get_bounding_box_from_address(address, offset=1):
    # Initialize geolocator
    geolocator = Nominatim(user_agent="topogreat")

    # Geocode the address to get latitude and longitude
    location = geolocator.geocode(address)
    
    if not location:
        raise ValueError(f"Address '{address}' could not be found.")

    lat, lon = location.latitude, location.longitude
    
    # Create a bounding box around the center (lat, lon)
    # Offset defines how far the bounding box will extend from the center (in degrees)
    # You can adjust this offset based on the area you're interested in
    bbox = (lon - offset, lat - offset, lon + offset, lat + offset)
    
    return bbox