"""
NASA POWER Data Fetching Module

This module provides functionality to fetch weather and climate data from the
NASA POWER (Prediction Of Worldwide Energy Resources) dataset via two methods:

1. **Zarr S3 Dataset** (get_data): 
   - Fast bulk access to historical data
   - Coarse grid: 0.5° lat × 0.625° lon resolution
   - Coordinates snap to nearest grid point (can be ~0.3° away)
   - Best for: Regional studies, bulk processing, offline access
   
2. **REST API** (get_data_from_api):
   - Fine resolution (~0.001°) matching native MERRA-2 grid
   - Coordinates match requests within 0.001°
   - Best for: Precise point queries, matching GPS coordinates
   - Requires network connection, subject to rate limits

⚠️ IMPORTANT: API and Zarr methods return DIFFERENT VALUES for the same location
because they sample different physical grid points. Choose one method and use it
consistently within your project.

See GRID_RESOLUTION_COMPARISON.md for detailed comparison and recommendations.
"""

import zarr
import s3fs
import fsspec
import xarray as xr
import numpy as np
import pandas as pd
import requests
import urllib3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import multiprocessing
from functools import partial
from dataclasses import dataclass

# Disable SSL warnings for multiprocessing API requests (following NASA POWER example)
urllib3.disable_warnings()


# ============================
# CONSTANTS
# ============================

VALID_FREQUENCIES = ['hourly', 'daily', 'monthly']
REF_DATE = datetime(1980, 1, 1)

# Common NASA POWER parameters
COMMON_PARAMETERS = {
    'T2M': 'Temperature at 2 Meters (°C)',
    'T2M_MAX': 'Maximum Temperature at 2 Meters (°C)',
    'T2M_MIN': 'Minimum Temperature at 2 Meters (°C)',
    'RH2M': 'Relative Humidity at 2 Meters (%)',
    'WS2M': 'Wind Speed at 2 Meters (m/s)',
    'PS': 'Surface Pressure (kPa)',
}


def _fetch_single_api_query(args: Tuple[int, Dict[str, Any], bool]) -> Tuple[int, Dict[str, Any]]:
    """
    Helper function for multiprocessing to fetch a single API query.
    
    This function is defined at module level because multiprocessing requires
    picklable functions (methods cannot be pickled).
    
    Args:
        args: Tuple of (index, query_dict, include_timing)
    
    Returns:
        Tuple of (index, result_dict)
    """
    import time
    
    index, query, include_timing = args
    start_time = time.time()
    
    try:
        # Extract query parameters
        start_date = query['start_date']
        end_date = query['end_date']
        latitude = query['latitude']
        longitude = query['longitude']
        parameter = query['parameter']
        frequency = query.get('frequency', 'daily')
        community = query.get('community', 'ag')
        
        # Validate frequency
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f'Invalid frequency "{frequency}". '
                f'Valid options: {", ".join(VALID_FREQUENCIES)}'
            )
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        
        if start_dt > end_dt:
            raise ValueError('start_date must be before or equal to end_date')
        
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= longitude <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        
        # Convert dates to appropriate format for API
        if frequency == 'monthly':
            start_str = start_dt.strftime("%Y")
            end_str = end_dt.strftime("%Y")
        else:
            start_str = start_dt.strftime("%Y%m%d")
            end_str = end_dt.strftime("%Y%m%d")
        
        # Build API URL
        base_url = "https://power.larc.nasa.gov/api/temporal"
        url = (
            f"{base_url}/{frequency}/point?"
            f"start={start_str}&end={end_str}&"
            f"latitude={latitude}&longitude={longitude}&"
            f"community={community}&parameters={parameter}&"
            f"format=json&header=true&time-standard=utc"
        )
        
        # Make API request
        response = requests.get(url, verify=False, timeout=30.0)
        response.raise_for_status()
        api_data = response.json()
        
        # Check for API errors
        if 'messages' in api_data and api_data['messages']:
            error_msg = '; '.join(api_data['messages'])
            raise Exception(f'API returned errors: {error_msg}')
        
        # Parse response
        if 'properties' not in api_data or 'parameter' not in api_data['properties']:
            raise Exception('Invalid API response structure')
        
        param_data = api_data['properties']['parameter'].get(parameter, {})
        
        # Extract parameter description
        param_info = api_data.get('parameters', {}).get(parameter, {})
        param_description = param_info.get('longname', COMMON_PARAMETERS.get(parameter, 'Unknown parameter'))
        
        # Get actual coordinates
        actual_coords = api_data['geometry']['coordinates']
        actual_lon = actual_coords[0]
        actual_lat = actual_coords[1]
        
        # Convert data to list of date-value pairs
        values = []
        for date_str, value in sorted(param_data.items()):
            # Skip annual averages for monthly data
            if frequency == 'monthly' and len(date_str) == 6 and date_str.endswith('13'):
                continue
            
            # Convert date string to standard format
            try:
                if frequency == 'monthly':
                    if len(date_str) == 6:
                        date_obj = datetime.strptime(date_str, "%Y%m")
                        formatted_date = date_obj.strftime("%Y-%m")
                    else:
                        formatted_date = date_str
                else:
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                formatted_date = date_str
            
            # Handle fill values
            fill_value = api_data.get('header', {}).get('fill_value', -999.0)
            if value == fill_value or (isinstance(value, float) and np.isnan(value)):
                value = None
            
            values.append({
                'date': formatted_date,
                'value': value
            })
        
        result = {
            'parameter': parameter,
            'parameter_description': param_description,
            'frequency': frequency,
            'latitude': actual_lat,
            'longitude': actual_lon,
            'data': values,
            'source': 'api_multiprocessing'
        }
        
        duration_ms = (time.time() - start_time) * 1000
        if include_timing:
            result['_timing_ms'] = round(duration_ms, 2)
        
        return (index, result)
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_result = {
            'error': str(e),
            'query': query
        }
        if include_timing:
            error_result['_timing_ms'] = round(duration_ms, 2)
        return (index, error_result)


def _fetch_single_zarr_query(args: Tuple[int, Dict[str, Any], bool]) -> Tuple[int, Dict[str, Any]]:
    """
    Helper function for multiprocessing to fetch a single Zarr query.
    
    This function is defined at module level because multiprocessing requires
    picklable functions (methods cannot be pickled).
    
    Args:
        args: Tuple of (index, query_dict, include_timing)
    
    Returns:
        Tuple of (index, result_dict)
    """
    import time
    
    index, query, include_timing = args
    start_time = time.time()
    
    try:
        # Extract query parameters
        start_date = query['start_date']
        end_date = query['end_date']
        latitude = query['latitude']
        longitude = query['longitude']
        parameter = query['parameter']
        frequency = query.get('frequency', 'daily')
        
        # Validate frequency
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f'Invalid frequency "{frequency}". '
                f'Valid options: {", ".join(VALID_FREQUENCIES)}'
            )
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        
        if start_dt > end_dt:
            raise ValueError('start_date must be before or equal to end_date')
        
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= longitude <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        
        # Build Zarr URL (using HTTPS endpoint)
        zarr_url = f"https://nasa-power.s3.us-west-2.amazonaws.com/merra2/temporal/power_merra2_{frequency}_temporal_utc.zarr"
        
        # Open dataset with xarray using fsspec mapper
        filepath_mapped = fsspec.get_mapper(zarr_url)
        ds = xr.open_zarr(filepath_mapped, consolidated=True)
        
        # Check if parameter exists
        if parameter not in ds:
            available = list(ds.data_vars.keys())
            raise KeyError(
                f'Parameter "{parameter}" not found in dataset. '
                f'Available parameters: {", ".join(available)}'
            )
        
        # Select data for the location (nearest grid point) and time range
        data_slice = ds[parameter].sel(
            lat=latitude,
            lon=longitude,
            method='nearest'
        ).sel(
            time=slice(start_dt, end_dt)
        ).load()
        
        # Get actual coordinates (after snapping to nearest grid point)
        actual_lat = float(data_slice.lat.values)
        actual_lon = float(data_slice.lon.values)
        
        # Convert to list of date-value pairs
        values = []
        
        # Handle single value case (when start_date == end_date and single time point)
        if data_slice.dims == ():
            # Scalar - single time point
            time_val = data_slice.time.values
            date_obj = pd.Timestamp(time_val).to_pydatetime()
            date_str = date_obj.strftime('%Y-%m-%d')
            
            value = float(data_slice.values)
            if np.isnan(value):
                value = None
            
            values.append({
                'date': date_str,
                'value': value
            })
        else:
            # Array of time points
            for i, time_val in enumerate(data_slice.time.values):
                date_obj = pd.Timestamp(time_val).to_pydatetime()
                date_str = date_obj.strftime('%Y-%m-%d')
                
                value = float(data_slice.values[i])
                if np.isnan(value):
                    value = None
                
                values.append({
                    'date': date_str,
                    'value': value
                })
        
        result = {
            'parameter': parameter,
            'parameter_description': COMMON_PARAMETERS.get(parameter, 'Unknown parameter'),
            'frequency': frequency,
            'latitude': actual_lat,
            'longitude': actual_lon,
            'data': values,
            'source': 'zarr_multiprocessing'
        }
        
        duration_ms = (time.time() - start_time) * 1000
        if include_timing:
            result['_timing_ms'] = round(duration_ms, 2)
        
        return (index, result)
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_result = {
            'error': str(e),
            'query': query
        }
        if include_timing:
            error_result['_timing_ms'] = round(duration_ms, 2)
        return (index, error_result)


class NASAPowerDataFetcher:
    """Fetches data from NASA POWER Zarr datasets on S3"""
    
    def __init__(self):
        """Initialize the S3 filesystem connection"""
        self.s3 = s3fs.S3FileSystem(anon=True)
        self._cache = {}  # Simple cache for opened datasets
    
    # def get_data(
    #     self,
    #     start_date: str,
    #     end_date: str,
    #     latitude: float,
    #     longitude: float,
    #     parameter: str,
    #     frequency: str = 'daily'
    # ) -> Dict[str, Any]:
    #     """
    #     Fetch NASA POWER data for a specific location and time range.
        
    #     Args:
    #         start_date: Start date in YYYY-MM-DD format
    #         end_date: End date in YYYY-MM-DD format
    #         latitude: Latitude coordinate (-90 to 90)
    #         longitude: Longitude coordinate (-180 to 180)
    #         parameter: Parameter name (e.g., T2M, RH2M, PRECTOTCORR)
    #         frequency: Data frequency - 'hourly', 'daily', or 'monthly'
        
    #     Returns:
    #         Dictionary containing the fetched data and metadata
        
    #     Raises:
    #         ValueError: If parameters are invalid
    #         KeyError: If parameter not found in dataset
    #         Exception: For other errors during data fetching
    #     """
    #     # Validate frequency
    #     if frequency not in VALID_FREQUENCIES:
    #         raise ValueError(
    #             f'Invalid frequency "{frequency}". '
    #             f'Valid options: {", ".join(VALID_FREQUENCIES)}'
    #         )
        
    #     try:
    #         start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    #         end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    #     except ValueError:
    #         raise ValueError('Dates must be in YYYY-MM-DD format')
        
    #     if start_dt > end_dt:
    #         raise ValueError('start_date must be before or equal to end_date')
        
    #     if not (-90 <= latitude <= 90):
    #         raise ValueError('Latitude must be between -90 and 90')
    #     if not (-180 <= longitude <= 180):
    #         raise ValueError('Longitude must be between -180 and 180')
        
    #     zarr_root = f"s3://nasa-power/merra2/temporal/power_merra2_{frequency}_temporal_utc.zarr"
        
    #     cache_key = f"{frequency}"
    #     if cache_key not in self._cache:
    #         store = s3fs.S3Map(root=zarr_root, s3=self.s3, check=False)
    #         root = zarr.open(store, mode='r')
    #         self._cache[cache_key] = root
    #     else:
    #         root = self._cache[cache_key]
        
    #     time_coord = root["time"][:]
    #     lat_coord = root["lat"][:]
    #     lon_coord = root["lon"][:]
        
    #     if parameter not in root:
    #         available = list(root.group_keys())
    #         raise KeyError(
    #             f'Parameter "{parameter}" not found in dataset. '
    #             f'Available parameters: {", ".join(available)}'
    #         )
        
    #     start_days = (start_dt - REF_DATE).days
    #     end_days = (end_dt - REF_DATE).days
        
    #     lat_idx = int(np.argmin(np.abs(lat_coord - latitude)))
    #     lon_idx = int(np.argmin(np.abs(lon_coord - longitude)))
        
    #     time_mask = (time_coord >= start_days) & (time_coord <= end_days)
    #     time_indices = np.where(time_mask)[0]
        
    #     if len(time_indices) == 0:
    #         raise ValueError(
    #             f'No data found for the specified date range: {start_date} to {end_date}'
    #         )
        
    #     data_array = root[parameter]
    #     values = []
        
    #     for time_idx in time_indices:
    #         value = float(data_array[time_idx, lat_idx, lon_idx])
    #         days_val = int(time_coord[time_idx])
    #         date_obj = REF_DATE + timedelta(days=days_val)
    #         date_str = date_obj.strftime('%Y-%m-%d')
            
    #         if np.isnan(value):
    #             value = None
            
    #         values.append({
    #             'date': date_str,
    #             'value': value
    #         })
        
    #     return {
    #         'parameter': parameter,
    #         'parameter_description': COMMON_PARAMETERS.get(parameter, 'Unknown parameter'),
    #         'frequency': frequency,
    #         'latitude': float(lat_coord[lat_idx]),
    #         'longitude': float(lon_coord[lon_idx]),
    #         'data': values
    #     }
    
    def get_data_from_api(
        self,
        start_date: str,
        end_date: str,
        latitude: float,
        longitude: float,
        parameter: str,
        frequency: str = 'daily',
        community: str = 'ag'
    ) -> Dict[str, Any]:
        """
        Fetch NASA POWER data from the REST API instead of Zarr.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            latitude: Latitude coordinate (-90 to 90)
            longitude: Longitude coordinate (-180 to 180)
            parameter: Parameter name (e.g., T2M, RH2M, PRECTOTCORR)
            frequency: Data frequency - 'hourly', 'daily', or 'monthly'
            community: Community parameter for API (default: 'ag')
        
        Returns:
            Dictionary containing the fetched data and metadata
        
        Raises:
            ValueError: If parameters are invalid
            requests.exceptions.RequestException: For API request errors
        """
        # Validate frequency
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f'Invalid frequency "{frequency}". '
                f'Valid options: {", ".join(VALID_FREQUENCIES)}'
            )
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        
        if start_dt > end_dt:
            raise ValueError('start_date must be before or equal to end_date')
        
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= longitude <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        
        # Convert dates to appropriate format for API
        # Monthly/Hourly use YYYY format, Daily uses YYYYMMDD format
        if frequency == 'monthly':
            start_str = start_dt.strftime("%Y")
            end_str = end_dt.strftime("%Y")
        else:
            start_str = start_dt.strftime("%Y%m%d")
            end_str = end_dt.strftime("%Y%m%d")
        
        # Build API URL
        base_url = "https://power.larc.nasa.gov/api/temporal"
        url = (
            f"{base_url}/{frequency}/point?"
            f"start={start_str}&end={end_str}&"
            f"latitude={latitude}&longitude={longitude}&"
            f"community={community}&parameters={parameter}&"
            f"format=json&header=true&time-standard=utc"
        )
        
        # Make API request
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            api_data = response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f'API request failed: {str(e)}')
        
        # Check for API errors
        if 'messages' in api_data and api_data['messages']:
            error_msg = '; '.join(api_data['messages'])
            raise Exception(f'API returned errors: {error_msg}')
        
        # Parse response and transform to match get_data() format
        if 'properties' not in api_data or 'parameter' not in api_data['properties']:
            raise Exception('Invalid API response structure')
        
        param_data = api_data['properties']['parameter'].get(parameter, {})
        
        # Extract parameter description from response
        param_info = api_data.get('parameters', {}).get(parameter, {})
        param_description = param_info.get('longname', COMMON_PARAMETERS.get(parameter, 'Unknown parameter'))
        
        # Get actual coordinates from response (API rounds to nearest grid point)
        actual_coords = api_data['geometry']['coordinates']
        actual_lon = actual_coords[0]
        actual_lat = actual_coords[1]
        
        # Convert data to list of date-value pairs
        values = []
        for date_str, value in sorted(param_data.items()):
            # Skip annual averages for monthly data (they end in "13", e.g., "200013")
            if frequency == 'monthly' and len(date_str) == 6 and date_str.endswith('13'):
                continue
            
            # Convert date string to standard format based on frequency
            try:
                if frequency == 'monthly':
                    # Monthly format: YYYYMM (e.g., "200001") -> YYYY-MM
                    if len(date_str) == 6:
                        date_obj = datetime.strptime(date_str, "%Y%m")
                        formatted_date = date_obj.strftime("%Y-%m")
                    else:
                        formatted_date = date_str
                else:
                    # Daily/Hourly format: YYYYMMDD -> YYYY-MM-DD
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                formatted_date = date_str
            
            # Handle fill values (-999.0) as None
            fill_value = api_data.get('header', {}).get('fill_value', -999.0)
            if value == fill_value or (isinstance(value, float) and np.isnan(value)):
                value = None
            
            values.append({
                'date': formatted_date,
                'value': value
            })
        
        return {
            'parameter': parameter,
            'parameter_description': param_description,
            'frequency': frequency,
            'latitude': actual_lat,
            'longitude': actual_lon,
            'data': values,
            'source': 'api'
        }
    
    def get_data_from_zarr_with_xarray(
        self,
        start_date: str,
        end_date: str,
        latitude: float,
        longitude: float,
        parameter: str,
        frequency: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Fetch NASA POWER data from Zarr using xarray.
        
        This method uses xarray for more convenient data access with the same
        Zarr dataset as get_data(). It provides a cleaner interface using
        xarray's label-based selection.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            latitude: Latitude coordinate (-90 to 90)
            longitude: Longitude coordinate (-180 to 180)
            parameter: Parameter name (e.g., T2M, RH2M, PRECTOTCORR)
            frequency: Data frequency - 'hourly', 'daily', or 'monthly'
        
        Returns:
            Dictionary containing the fetched data and metadata
        
        Raises:
            ValueError: If parameters are invalid
            KeyError: If parameter not found in dataset
        """
        # Validate frequency
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f'Invalid frequency "{frequency}". '
                f'Valid options: {", ".join(VALID_FREQUENCIES)}'
            )
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        
        if start_dt > end_dt:
            raise ValueError('start_date must be before or equal to end_date')
        
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= longitude <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        
        # Build Zarr URL (using HTTPS endpoint)
        zarr_url = f"https://nasa-power.s3.us-west-2.amazonaws.com/merra2/temporal/power_merra2_{frequency}_temporal_utc.zarr"
        
        # Open dataset with xarray using fsspec mapper
        filepath_mapped = fsspec.get_mapper(zarr_url)
        ds = xr.open_zarr(filepath_mapped, consolidated=True)
        
        # Check if parameter exists
        if parameter not in ds:
            available = list(ds.data_vars.keys())
            raise KeyError(
                f'Parameter "{parameter}" not found in dataset. '
                f'Available parameters: {", ".join(available)}'
            )
        
        # Select data for the location (nearest grid point) and time range
        # First select nearest lat/lon, then slice time
        data_slice = ds[parameter].sel(
            lat=latitude,
            lon=longitude,
            method='nearest'
        ).sel(
            time=slice(start_dt, end_dt)
        ).load()
        
        # Get actual coordinates (after snapping to nearest grid point)
        actual_lat = float(data_slice.lat.values)
        actual_lon = float(data_slice.lon.values)
        
        # Convert to list of date-value pairs
        values = []
        
        # Handle single value case (when start_date == end_date and single time point)
        if data_slice.dims == ():
            # Scalar - single time point
            time_val = data_slice.time.values
            date_obj = pd.Timestamp(time_val).to_pydatetime()
            date_str = date_obj.strftime('%Y-%m-%d')
            
            value = float(data_slice.values)
            if np.isnan(value):
                value = None
            
            values.append({
                'date': date_str,
                'value': value
            })
        else:
            # Array of time points
            for i, time_val in enumerate(data_slice.time.values):
                # Convert numpy datetime64 to Python datetime
                date_obj = pd.Timestamp(time_val).to_pydatetime()
                date_str = date_obj.strftime('%Y-%m-%d')
                
                value = float(data_slice.values[i])
                if np.isnan(value):
                    value = None
                
                values.append({
                    'date': date_str,
                    'value': value
                })
        
        return {
            'parameter': parameter,
            'parameter_description': COMMON_PARAMETERS.get(parameter, 'Unknown parameter'),
            'frequency': frequency,
            'latitude': actual_lat,
            'longitude': actual_lon,
            'data': values,
            'source': 'zarr_xarray'
        }
    
    async def get_data_from_zarr_batch_async(
        self,
        queries: List[Dict[str, Any]],
        max_concurrent: int = 10,
        include_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch NASA POWER data for multiple locations/dates concurrently.
        
        This method processes multiple queries in parallel using asyncio,
        with a configurable concurrency limit (default 10 concurrent requests).
        
        Args:
            queries: List of query dictionaries, each containing:
                - start_date: Start date in YYYY-MM-DD format
                - end_date: End date in YYYY-MM-DD format
                - latitude: Latitude coordinate (-90 to 90)
                - longitude: Longitude coordinate (-180 to 180)
                - parameter: Parameter name (e.g., T2M, RH2M)
                - frequency: Data frequency (optional, default: 'daily')
            max_concurrent: Maximum number of concurrent requests (default: 10)
            include_timing: If True, adds '_timing_ms' field to each result
        
        Returns:
            List of result dictionaries in the same order as input queries.
            Each result contains either:
                - The weather data (same format as get_data_from_zarr_with_xarray)
                - An error dict with 'error' key if the query failed
        
        Example:
            queries = [
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            ]
            results = await fetcher.get_data_from_zarr_batch_async(queries)
        """
        import time
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_single(query: Dict[str, Any], index: int) -> tuple[int, Dict[str, Any]]:
            """Fetch a single query with semaphore control."""
            async with semaphore:
                start_time = time.time()
                try:
                    # Run the blocking xarray call in a thread pool
                    result = await asyncio.to_thread(
                        self.get_data_from_zarr_with_xarray,
                        start_date=query['start_date'],
                        end_date=query['end_date'],
                        latitude=query['latitude'],
                        longitude=query['longitude'],
                        parameter=query['parameter'],
                        frequency=query.get('frequency', 'daily')
                    )
                    duration_ms = (time.time() - start_time) * 1000
                    if include_timing:
                        result['_timing_ms'] = round(duration_ms, 2)
                    return (index, result)
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    error_result = {
                        'error': str(e),
                        'query': query
                    }
                    if include_timing:
                        error_result['_timing_ms'] = round(duration_ms, 2)
                    return (index, error_result)
        
        # Create tasks for all queries
        tasks = [fetch_single(query, i) for i, query in enumerate(queries)]
        
        # Run all tasks concurrently
        indexed_results = await asyncio.gather(*tasks)
        
        # Sort by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract just the results (without index)
        return [result for _, result in indexed_results]
    
    def get_data_from_zarr_batch_multiprocessing(
        self,
        queries: List[Dict[str, Any]],
        max_processes: int = 5,
        include_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch NASA POWER data from Zarr for multiple locations/dates using multiprocessing.
        
        This method processes multiple queries in parallel using Python's multiprocessing,
        with a default of 5 concurrent processes.
        
        Note: Uses 'spawn' start method to avoid deadlocks with xarray/fsspec.
        
        Args:
            queries: List of query dictionaries, each containing:
                - start_date: Start date in YYYY-MM-DD format
                - end_date: End date in YYYY-MM-DD format
                - latitude: Latitude coordinate (-90 to 90)
                - longitude: Longitude coordinate (-180 to 180)
                - parameter: Parameter name (e.g., T2M, RH2M)
                - frequency: Data frequency (optional, default: 'daily')
            max_processes: Maximum number of concurrent processes (default: 5)
            include_timing: If True, adds '_timing_ms' field to each result
        
        Returns:
            List of result dictionaries in the same order as input queries.
            Each result contains either:
                - The weather data (same format as get_data_from_zarr_with_xarray)
                - An error dict with 'error' key if the query failed
        
        Example:
            queries = [
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            ]
            results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
        """
        # Prepare indexed queries for maintaining order
        indexed_queries = [(i, query, include_timing) for i, query in enumerate(queries)]
        
        # Use 'spawn' context to avoid deadlocks with xarray/fsspec
        # 'fork' can cause issues with remote file systems
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=max_processes) as pool:
            indexed_results = pool.map(_fetch_single_zarr_query, indexed_queries)
        
        # Sort by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract just the results (without index)
        return [result for _, result in indexed_results]
    
    async def get_data_from_api_batch_async(
        self,
        queries: List[Dict[str, Any]],
        max_concurrent: int = 10,
        include_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch NASA POWER data from REST API for multiple locations/dates concurrently.
        
        This method processes multiple API queries in parallel using asyncio,
        with a configurable concurrency limit (default 10 concurrent requests).
        
        Note: Be mindful of NASA POWER API rate limits when setting max_concurrent.
        
        Args:
            queries: List of query dictionaries, each containing:
                - start_date: Start date in YYYY-MM-DD format
                - end_date: End date in YYYY-MM-DD format
                - latitude: Latitude coordinate (-90 to 90)
                - longitude: Longitude coordinate (-180 to 180)
                - parameter: Parameter name (e.g., T2M, RH2M)
                - frequency: Data frequency (optional, default: 'daily')
                - community: Community parameter (optional, default: 'ag')
            max_concurrent: Maximum number of concurrent requests (default: 10)
            include_timing: If True, adds '_timing_ms' field to each result
        
        Returns:
            List of result dictionaries in the same order as input queries.
            Each result contains either:
                - The weather data (same format as get_data_from_api)
                - An error dict with 'error' key if the query failed
        
        Example:
            queries = [
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            ]
            results = await fetcher.get_data_from_api_batch_async(queries)
        """
        import time
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_single(query: Dict[str, Any], index: int) -> tuple[int, Dict[str, Any]]:
            """Fetch a single API query with semaphore control."""
            async with semaphore:
                start_time = time.time()
                try:
                    # Run the blocking API call in a thread pool
                    result = await asyncio.to_thread(
                        self.get_data_from_api,
                        start_date=query['start_date'],
                        end_date=query['end_date'],
                        latitude=query['latitude'],
                        longitude=query['longitude'],
                        parameter=query['parameter'],
                        frequency=query.get('frequency', 'daily'),
                        community=query.get('community', 'ag')
                    )
                    duration_ms = (time.time() - start_time) * 1000
                    if include_timing:
                        result['_timing_ms'] = round(duration_ms, 2)
                    return (index, result)
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    error_result = {
                        'error': str(e),
                        'query': query
                    }
                    if include_timing:
                        error_result['_timing_ms'] = round(duration_ms, 2)
                    return (index, error_result)
        
        # Create tasks for all queries
        tasks = [fetch_single(query, i) for i, query in enumerate(queries)]
        
        # Run all tasks concurrently
        indexed_results = await asyncio.gather(*tasks)
        
        # Sort by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract just the results (without index)
        return [result for _, result in indexed_results]
    
    def get_data_from_api_batch_multiprocessing(
        self,
        queries: List[Dict[str, Any]],
        max_processes: int = 5,
        include_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch NASA POWER data from REST API for multiple locations/dates using multiprocessing.
        
        This method processes multiple API queries in parallel using Python's multiprocessing,
        following NASA POWER's recommended approach with a default of 5 concurrent processes.
        
        Note: NASA POWER recommends not exceeding 5 concurrent requests.
        Note: Uses 'spawn' start method to avoid deadlock warnings in multi-threaded environments.
        
        Args:
            queries: List of query dictionaries, each containing:
                - start_date: Start date in YYYY-MM-DD format
                - end_date: End date in YYYY-MM-DD format
                - latitude: Latitude coordinate (-90 to 90)
                - longitude: Longitude coordinate (-180 to 180)
                - parameter: Parameter name (e.g., T2M, RH2M)
                - frequency: Data frequency (optional, default: 'daily')
                - community: Community parameter (optional, default: 'ag')
            max_processes: Maximum number of concurrent processes (default: 5, NASA recommended limit)
            include_timing: If True, adds '_timing_ms' field to each result
        
        Returns:
            List of result dictionaries in the same order as input queries.
            Each result contains either:
                - The weather data (same format as get_data_from_api)
                - An error dict with 'error' key if the query failed
        
        Example:
            queries = [
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
                {'start_date': '2024-06-01', 'end_date': '2024-06-01',
                 'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            ]
            results = fetcher.get_data_from_api_batch_multiprocessing(queries)
        """
        # Prepare indexed queries for maintaining order
        indexed_queries = [(i, query, include_timing) for i, query in enumerate(queries)]
        
        # Use 'spawn' context to avoid fork() warnings in multi-threaded environments
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=max_processes) as pool:
            indexed_results = pool.map(_fetch_single_api_query, indexed_queries)
        
        # Sort by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract just the results (without index)
        return [result for _, result in indexed_results]
    
    def get_available_parameters(self, frequency: str = 'daily') -> List[str]:
        """
        Get list of available parameters in the dataset.
        
        Args:
            frequency: Data frequency - 'hourly', 'daily', or 'monthly'
        
        Returns:
            List of available parameter names
        """
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f'Invalid frequency "{frequency}". '
                f'Valid options: {", ".join(VALID_FREQUENCIES)}'
            )
        
        zarr_root = f"s3://nasa-power/merra2/spatial/power_merra2_{frequency}_spatial_utc.zarr"
        
        cache_key = f"{frequency}"
        if cache_key not in self._cache:
            store = s3fs.S3Map(root=zarr_root, s3=self.s3, check=False)
            root = zarr.open(store, mode='r')
            self._cache[cache_key] = root
        else:
            root = self._cache[cache_key]
        
        return list(root.group_keys())
    
    def get_parameter_info(self) -> Dict[str, str]:
        """
        Get information about common NASA POWER parameters.
        
        Returns:
            Dictionary mapping parameter names to descriptions
        """
        return COMMON_PARAMETERS.copy()


def enrich_locations_with_nasa_data(
    locations: List[Dict[str, Any]],
    parameters: List[str] = None,
    date_range_days: int = 1,
    frequency: str = 'daily'
) -> List[Dict[str, Any]]:
    """
    Enrich a list of location records with NASA POWER data.
    
    This function takes an array of records from iChatBio with eventDate, 
    decimalLatitude, and decimalLongitude, and enriches each record with
    NASA POWER weather data.
    
    Args:
        locations: List of location records with keys:
            - eventDate: Date/datetime string (ISO format) or None
            - decimalLatitude: Latitude coordinate or None
            - decimalLongitude: Longitude coordinate or None
        parameters: List of NASA POWER parameters to fetch (default: ['T2M'])
        date_range_days: Number of days to fetch (default: 1 = exact event date only)
                        Use values > 1 to fetch a range around the event date
        frequency: Data frequency - 'hourly', 'daily', or 'monthly' (default: 'daily')
    
    Returns:
        List of enriched location records with added 'nasaPowerProperties' field
    """
    if parameters is None:
        parameters = ['T2M']  # Default to temperature
    
    fetcher = NASAPowerDataFetcher()
    
    # Prepare all queries for batch processing
    queries = []
    location_query_map = []  # Maps query index to (location_index, parameter_index)
    location_validity = []  # Track which locations are valid for processing
    
    # First pass: validate locations and prepare queries
    for loc_idx, location in enumerate(locations):
        # Skip if essential data is missing
        event_date = location.get('eventDate')
        latitude = location.get('decimalLatitude')
        longitude = location.get('decimalLongitude')
        
        if event_date is None or latitude is None or longitude is None:
            location_validity.append(False)
            continue
        
        # Parse the event date (handle various ISO formats)
        try:
            # Extract just the date part from ISO datetime
            if 'T' in event_date:
                date_str = event_date.split('T')[0]
            else:
                date_str = event_date
            
            center_date = datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, AttributeError):
            # Invalid date format
            location_validity.append(False)
            continue
        
        # Location is valid
        location_validity.append(True)
        
        # Calculate date range
        if date_range_days == 1:
            # Fetch only the exact event date
            start_date = center_date
            end_date = center_date
        else:
            # Fetch a range around the event date
            half_range = date_range_days // 2
            start_date = center_date - timedelta(days=half_range)
            end_date = center_date + timedelta(days=half_range)
        
        # Create queries for each parameter
        for param_idx, parameter in enumerate(parameters):
            queries.append({
                'start_date': start_date.strftime("%Y-%m-%d"),
                'end_date': end_date.strftime("%Y-%m-%d"),
                'latitude': float(latitude),
                'longitude': float(longitude),
                'parameter': parameter,
                'frequency': frequency
            })
            location_query_map.append((loc_idx, param_idx))
    
    # Batch process all queries
    if queries:
        results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
    else:
        results = []
    
    # Group results back by location
    location_results = {}  # Maps location index to list of parameter results
    
    for query_idx, (loc_idx, param_idx) in enumerate(location_query_map):
        if loc_idx not in location_results:
            location_results[loc_idx] = []
        
        if query_idx < len(results):
            result = results[query_idx]
            if 'error' in result:
                # If there was an error, create error dict
                location_results[loc_idx].append({
                    'parameter': queries[query_idx]['parameter'],
                    'error': result['error']
                })
            else:
                # Successful result
                location_results[loc_idx].append(result)
        else:
            # Missing result (shouldn't happen, but handle gracefully)
            location_results[loc_idx].append({
                'parameter': queries[query_idx]['parameter'],
                'error': 'No result returned'
            })
    
    # Build enriched locations in order
    enriched_locations = []
    for loc_idx, location in enumerate(locations):
        enriched_record = location.copy()
        
        if location_validity[loc_idx] and loc_idx in location_results:
            # This location had valid data and was processed
            enriched_record['nasaPowerProperties'] = location_results[loc_idx]
        else:
            # This location was skipped (invalid data)
            enriched_record['nasaPowerProperties'] = None
        
        enriched_locations.append(enriched_record)
    
    return enriched_locations


# Convenience function for direct usage
# def fetch_nasa_power_data(
#     start_date: str,
#     end_date: str,
#     latitude: float,
#     longitude: float,
#     parameter: str,
#     frequency: str = 'daily'
# ) -> Dict[str, Any]:
#     """
#     Convenience function to fetch NASA POWER data from Zarr.
    
#     See NASAPowerDataFetcher.get_data() for parameter details.
#     """
#     fetcher = NASAPowerDataFetcher()
#     return fetcher.get_data(start_date, end_date, latitude, longitude, parameter, frequency)


def fetch_nasa_power_data_from_api(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    parameter: str,
    frequency: str = 'daily',
    community: str = 'ag'
) -> Dict[str, Any]:
    """
    Convenience function to fetch NASA POWER data from REST API.
    
    See NASAPowerDataFetcher.get_data_from_api() for parameter details.
    """
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_api(start_date, end_date, latitude, longitude, parameter, frequency, community)


def fetch_nasa_power_data_from_zarr_with_xarray(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    parameter: str,
    frequency: str = 'daily'
) -> Dict[str, Any]:
    """
    Convenience function to fetch NASA POWER data from Zarr using xarray.
    
    See NASAPowerDataFetcher.get_data_from_zarr_with_xarray() for parameter details.
    """
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_zarr_with_xarray(start_date, end_date, latitude, longitude, parameter, frequency)


async def fetch_nasa_power_data_batch_async(
    queries: List[Dict[str, Any]],
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch NASA POWER data for multiple queries concurrently.
    
    Args:
        queries: List of query dictionaries, each containing:
            - start_date: Start date in YYYY-MM-DD format
            - end_date: End date in YYYY-MM-DD format
            - latitude: Latitude coordinate (-90 to 90)
            - longitude: Longitude coordinate (-180 to 180)
            - parameter: Parameter name (e.g., T2M, RH2M)
            - frequency: Data frequency (optional, default: 'daily')
        max_concurrent: Maximum number of concurrent requests (default: 10)
    
    Returns:
        List of result dictionaries in the same order as input queries.
    
    Example:
        import asyncio
        
        queries = [
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': -33.8688, 'longitude': 151.2093, 'parameter': 'T2M'},
        ]
        
        results = asyncio.run(fetch_nasa_power_data_batch_async(queries))
    """
    fetcher = NASAPowerDataFetcher()
    return await fetcher.get_data_from_zarr_batch_async(queries, max_concurrent)


async def fetch_nasa_power_data_from_api_batch_async(
    queries: List[Dict[str, Any]],
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch NASA POWER data from API for multiple queries concurrently.
    
    Args:
        queries: List of query dictionaries, each containing:
            - start_date: Start date in YYYY-MM-DD format
            - end_date: End date in YYYY-MM-DD format
            - latitude: Latitude coordinate (-90 to 90)
            - longitude: Longitude coordinate (-180 to 180)
            - parameter: Parameter name (e.g., T2M, RH2M)
            - frequency: Data frequency (optional, default: 'daily')
            - community: Community parameter (optional, default: 'ag')
        max_concurrent: Maximum number of concurrent requests (default: 10)
    
    Returns:
        List of result dictionaries in the same order as input queries.
    
    Example:
        import asyncio
        
        queries = [
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': -33.8688, 'longitude': 151.2093, 'parameter': 'T2M'},
        ]
        
        results = asyncio.run(fetch_nasa_power_data_from_api_batch_async(queries))
    """
    fetcher = NASAPowerDataFetcher()
    return await fetcher.get_data_from_api_batch_async(queries, max_concurrent)


def fetch_nasa_power_data_from_api_batch_multiprocessing(
    queries: List[Dict[str, Any]],
    max_processes: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch NASA POWER data from API using multiprocessing.
    
    This follows NASA POWER's recommended approach for multi-point downloads,
    using Python's multiprocessing with a default of 5 concurrent processes
    (NASA's recommended limit).
    
    Args:
        queries: List of query dictionaries, each containing:
            - start_date: Start date in YYYY-MM-DD format
            - end_date: End date in YYYY-MM-DD format
            - latitude: Latitude coordinate (-90 to 90)
            - longitude: Longitude coordinate (-180 to 180)
            - parameter: Parameter name (e.g., T2M, RH2M)
            - frequency: Data frequency (optional, default: 'daily')
            - community: Community parameter (optional, default: 'ag')
        max_processes: Maximum number of concurrent processes (default: 5)
    
    Returns:
        List of result dictionaries in the same order as input queries.
    
    Example:
        queries = [
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': -33.8688, 'longitude': 151.2093, 'parameter': 'T2M'},
        ]
        
        results = fetch_nasa_power_data_from_api_batch_multiprocessing(queries)
    """
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_api_batch_multiprocessing(queries, max_processes)


def fetch_nasa_power_data_from_zarr_batch_multiprocessing(
    queries: List[Dict[str, Any]],
    max_processes: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch NASA POWER data from Zarr using multiprocessing.
    
    Args:
        queries: List of query dictionaries, each containing:
            - start_date: Start date in YYYY-MM-DD format
            - end_date: End date in YYYY-MM-DD format
            - latitude: Latitude coordinate (-90 to 90)
            - longitude: Longitude coordinate (-180 to 180)
            - parameter: Parameter name (e.g., T2M, RH2M)
            - frequency: Data frequency (optional, default: 'daily')
        max_processes: Maximum number of concurrent processes (default: 5)
    
    Returns:
        List of result dictionaries in the same order as input queries.
    
    Example:
        queries = [
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 40.7128, 'longitude': -74.0060, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': 35.6762, 'longitude': 139.6503, 'parameter': 'T2M'},
            {'start_date': '2024-06-01', 'end_date': '2024-06-01',
             'latitude': -33.8688, 'longitude': 151.2093, 'parameter': 'T2M'},
        ]
        
        results = fetch_nasa_power_data_from_zarr_batch_multiprocessing(queries)
    """
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_zarr_batch_multiprocessing(queries, max_processes)

