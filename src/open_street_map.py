"""
OpenStreetMap Nominatim Geocoding Module

This module provides functionality to geocode addresses/locations using
OpenStreetMap's Nominatim API to convert text addresses to latitude/longitude coordinates.
"""

import requests
from typing import Optional, Tuple
import time


def geocode_address(address: str, timeout: float = 10.0) -> Optional[Tuple[float, float]]:
    """
    Geocode an address/location string to latitude and longitude using OpenStreetMap Nominatim API.
    """
    if not address or not address.strip():
        return None
    
    # Build Nominatim API URL
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address.strip(),
        'format': 'json',
        'limit': 1  # Only need the first/best result
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
        
        # Extract lat/lon from first result
        first_result = results[0]
        lat_str = first_result.get('lat')
        lon_str = first_result.get('lon')
        
        if lat_str is None or lon_str is None:
            return None
        
        try:
            latitude = float(lat_str)
            longitude = float(lon_str)
            return (latitude, longitude)
        except (ValueError, TypeError):
            return None
            
    except requests.exceptions.RequestException:
        # Return None on network/API errors (caller can handle)
        return None
    except (KeyError, ValueError, TypeError):
        # Return None on parsing errors
        return None
