"""
OpenStreetMap Nominatim Geocoding Module

This module provides functionality to geocode addresses/locations using
OpenStreetMap's Nominatim API to convert text addresses to latitude/longitude coordinates.
"""

import requests
from typing import Optional, List, Dict


def geocode_address(address: str, timeout: float = 10.0) -> Optional[List[Dict]]:
    """
    Geocode an address/location string to latitude and longitude using OpenStreetMap Nominatim API.
    
    Returns a list of all matching results, each containing:
    - 'display_name': Full display name of the location
    - 'lat': Latitude as string
    - 'lon': Longitude as string
    - 'latitude': Latitude as float
    - 'longitude': Longitude as float
    - Other fields from Nominatim API response
    
    Returns None if no results found or on error.
    """
    if not address or not address.strip():
        return None
    
    # Build Nominatim API URL
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address.strip(),
        'format': 'json'
        # No limit - return all results so user can select
    }
    
    # Add User-Agent header (required by Nominatim usage policy)
    headers = {
        'User-Agent': 'iChatBio-NASA-POWER-Agent/1.0'
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        results = response.json()
        
        if not results or len(results) == 0:
            return None
        
        # Process all results and add parsed lat/lon as floats
        processed_results = []
        for result in results:
            lat_str = result.get('lat')
            lon_str = result.get('lon')
            
            if lat_str is not None and lon_str is not None:
                try:
                    latitude = float(lat_str)
                    longitude = float(lon_str)
                    # Add parsed coordinates to result dict
                    result_copy = result.copy()
                    result_copy['latitude'] = latitude
                    result_copy['longitude'] = longitude
                    processed_results.append(result_copy)
                except (ValueError, TypeError):
                    # Skip results with invalid coordinates
                    continue
        
        if not processed_results:
            return None
        
        return processed_results
            
    except requests.exceptions.RequestException:
        # Return None on network/API errors (caller can handle)
        return None
    except (KeyError, ValueError, TypeError):
        # Return None on parsing errors
        return None
