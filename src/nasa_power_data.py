"""
NASA POWER Data Fetching Module

This module provides functionality to fetch weather and climate data from the
NASA POWER (Prediction Of Worldwide Energy Resources) dataset
"""

import zarr
import s3fs
import fsspec
import xarray as xr
import numpy as np
import pandas as pd
import requests
import urllib3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import multiprocessing
from functools import partial
from dataclasses import dataclass
from ichatbio.agent_response import IChatBioAgentProcess

# Disable SSL warnings for multiprocessing API requests (following NASA POWER example)
urllib3.disable_warnings()


# ============================
# CONSTANTS
# ============================

VALID_FREQUENCIES = ['hourly', 'daily', 'monthly']
VALID_TEMPORALS = ['temporal', 'spatial']
VALID_TIMES = ['utc', 'lst']
REF_DATE = datetime(1980, 1, 1)

# Common NASA POWER parameters (fallback)
COMMON_PARAMETERS = {
    'T2M': 'Temperature at 2 Meters (°C)',
    'T2M_MAX': 'Maximum Temperature at 2 Meters (°C)',
    'T2M_MIN': 'Minimum Temperature at 2 Meters (°C)',
    'RH2M': 'Relative Humidity at 2 Meters (%)',
    'WS2M': 'Wind Speed at 2 Meters (m/s)',
    'PS': 'Surface Pressure (kPa)',
}

# Cache for parameter metadata from API
_PARAMETER_INFO_CACHE = None

def _get_parameter_description(parameter: str) -> str:
    global _PARAMETER_INFO_CACHE
    
    # Try to fetch from API if cache is empty
    if _PARAMETER_INFO_CACHE is None:
        try:
            url = "https://power.larc.nasa.gov/api/website/metadata/parameters"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            parameters_data = response.json()
            
            # Build cache
            _PARAMETER_INFO_CACHE = {}
            for param in parameters_data:
                param_id = param.get('id')
                param_name = param.get('name', 'Unknown parameter')
                if param_id:
                    _PARAMETER_INFO_CACHE[param_id] = param_name
        except Exception:
            # Fallback to COMMON_PARAMETERS
            _PARAMETER_INFO_CACHE = COMMON_PARAMETERS.copy()
    
    # Return from cache or fallback
    return _PARAMETER_INFO_CACHE.get(parameter, COMMON_PARAMETERS.get(parameter, 'Unknown parameter'))


def _fetch_single_api_query(args: Tuple[int, Dict[str, Any], bool]) -> Tuple[int, Dict[str, Any]]:
    import time
    
    index, query, include_timing = args
    start_time = time.time()
    api_url: Optional[str] = None
    
    try:
        start_date = query['start_date']
        end_date = query['start_date']
        latitude = query['latitude']
        longitude = query['longitude']
        parameter = query['parameter']
        frequency = query['frequency']
        community = query.get('community', 'ag')
        
        # Convert dates to appropriate format for API
        if frequency == 'monthly':
            start_str = start_date.strftime("%Y")
            end_str = end_date.strftime("%Y")
        else:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
        
        # Build API URL
        base_url = "https://power.larc.nasa.gov/api/temporal"
        api_url = (
            f"{base_url}/{frequency}/point?"
            f"start={start_str}&end={end_str}&"
            f"latitude={latitude}&longitude={longitude}&"
            f"community={community}&parameters={parameter}&"
            f"format=json&header=true&time-standard=utc"
        )
        
        # Make API request
        response = requests.get(api_url, verify=False, timeout=30.0)
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
            'error': {
                'api_url': api_url,
                'latitude': query.get('latitude'),
                'longitude': query.get('longitude'),
                'start_date': query.get('start_date'),
                'end_date': query.get('end_date'),
                'message': str(e),
            }
        }
        if include_timing:
            error_result['_timing_ms'] = round(duration_ms, 2)
        return (index, error_result)


def _fetch_single_zarr_query(args: Tuple[int, Dict[str, Any], bool]) -> Tuple[int, Dict[str, Any]]:
    import time
    
    index, query, include_timing = args
    start_time = time.time()
    zarr_url: Optional[str] = None
    
    try:
        # Extract query parameters
        start_date = query['start_date']
        end_date = query['end_date']
        latitude = query['latitude']
        longitude = query['longitude']
        parameter = query['parameter']
        frequency = query['frequency']
        source = query['source']
        temporal = query['temporal']
        time_standard = query['time_standard']

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        print(f"start_dt: {start_dt}, end_dt: {end_dt}")
        
        # Build Zarr URL dynamically based on source, frequency, temporal, and time
        zarr_url = f"https://nasa-power.s3.us-west-2.amazonaws.com/{source}/{temporal}/power_{source}_{frequency}_{temporal}_{time_standard}.zarr"
        
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
                
                if value is not None:
                    values.append({
                        'date': date_str,
                        'value': value
                    })
                else:
                    values.append({
                        'error': f'No data found for date: {date_str}'
                    })
        
        result = {
            'parameter': parameter,
            'latitude': actual_lat,
            'longitude': actual_lon,
            'data': values
        }
        
        duration_ms = (time.time() - start_time) * 1000
        if include_timing:
            result['_timing_ms'] = round(duration_ms, 2)
        
        return (index, result)
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_result = {
            'error': {
                'zarr_url': zarr_url,
                'latitude': query.get('latitude'),
                'longitude': query.get('longitude'),
                'start_date': query.get('start_date'),
                'end_date': query.get('end_date'),
                'message': str(e),
            }
        }
        if include_timing:
            error_result['_timing_ms'] = round(duration_ms, 2)
        return (index, error_result)


class NASAPowerDataFetcher:
    """Fetches data from NASA POWER Zarr datasets on S3"""
    
    def __init__(self):
        """Initialize the S3 filesystem connection"""
        self.s3 = s3fs.S3FileSystem(anon=True)
        self._cache = {}
    
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
        frequency: str = 'daily',
        source: str = 'merra2',
        temporal: str = 'temporal',
        time: str = 'utc'
    ) -> Dict[str, Any]:

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
        
        # Build Zarr URL dynamically based on source, frequency, temporal, and time
        zarr_url = f"https://nasa-power.s3.us-west-2.amazonaws.com/{source}/{temporal}/power_{source}_{frequency}_{temporal}_{time}.zarr"
        
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
        
        # Select data for the location (nearest grid point) and time range.
        # First select nearest lat/lon, then slice time. start_dt/end_dt from "YYYY-MM-DD"; inclusive.
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
        
        # Get parameter description from API
        param_info = self.get_parameter_info()
        param_description = param_info.get(parameter, 'Unknown parameter')
        
        return {
            'parameter': parameter,
            'parameter_description': param_description,
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
                        frequency=query.get('frequency', 'daily'),
                        source=query.get('source', 'merra2'),
                        temporal=query.get('temporal', 'temporal'),
                        time=query.get('time', 'utc')
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
        Get information about NASA POWER parameters from the API metadata endpoint.
        
        Returns:
            Dictionary mapping parameter IDs to their full names
        """
        try:
            # Fetch parameters from NASA POWER metadata API
            url = "https://power.larc.nasa.gov/api/website/metadata/parameters"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            parameters_data = response.json()
            
            # Convert to dict mapping id -> name
            param_info = {}
            for param in parameters_data:
                param_id = param.get('id')
                param_name = param.get('name', 'Unknown parameter')
                if param_id:
                    param_info[param_id] = param_name
            
            return param_info
        except Exception as e:
            # Fallback to COMMON_PARAMETERS if API fails
            return COMMON_PARAMETERS.copy()
    
    def get_parameter_metadata(self) -> List[Dict[str, Any]]:
        try:
            url = "https://power.larc.nasa.gov/api/website/metadata/parameters"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return []


def _nasa_properties_has_any_data(props: Optional[List[Dict[str, Any]]]) -> bool:
    """Return True if nasaPowerProperties has at least one non-empty data array."""
    if not props or not isinstance(props, list):
        return False
    for param_block in props:
        if isinstance(param_block, dict) and (param_block.get("data") or []):
            return True
    return False


def _count_valid_nasa_date_rows(enriched_locations: List[Dict[str, Any]]) -> int:
    """Count valid time-series rows (date/value entries) under nasaPowerProperties."""
    total = 0
    for loc in enriched_locations:
        props = loc.get("nasaPowerProperties")
        if props is None:
            continue
        blocks = props if isinstance(props, list) else [props]
        for block in blocks:
            if isinstance(block, dict):
                data = block.get("data")
                if isinstance(data, list):
                    total += len(data)
    return total


async def enrich_locations_with_nasa_data(
    locations: List[Dict[str, Any]],
    parameters: List[str] = None,
    frequency: str = 'daily',
    source: str = 'merra2',
    temporal: str = 'temporal',
    time_standard: str = 'utc',
    process: IChatBioAgentProcess = None
) -> List[Dict[str, Any]]:
    if parameters is None:
        parameters = ['T2M']

    fetcher = NASAPowerDataFetcher()

    if frequency not in VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency: {frequency}")
    if temporal not in VALID_TEMPORALS:
        raise ValueError(f"Invalid temporal: {temporal}")
    if time_standard not in VALID_TIMES:
        raise ValueError(f"Invalid time_standard: {time_standard}")

    queries: List[Dict[str, Any]] = []

    for loc_idx, location in enumerate(locations):
        start_date = location.get('startDate')
        end_date = location.get('endDate')
        latitude = location.get('decimalLatitude')
        longitude = location.get('decimalLongitude')

        for parameter in parameters:
            queries.append(
                {
                    'start_date': start_date,
                    'end_date': end_date,
                    'latitude': float(latitude),
                    'longitude': float(longitude),
                    'parameter': parameter,
                    'frequency': frequency,
                    'source': source,
                    'temporal': temporal,
                    'time_standard': time_standard,
                }
            )

    results = fetcher.get_data_from_zarr_batch_multiprocessing(queries) if queries else []
    # await process.log(f"NASA POWER Zarr query results: {results}")

    final_results: List[Dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict) and 'error' in item:
            err = item.get('error')
            try:
                message = err.get('message')
                await process.log(f"NASA POWER get data error: {message}")
            except Exception:
                pass
            continue

        if isinstance(item, dict) and 'data' in item:
            data = item.get('data')
            clean_data = []
            for data_item in data or []:
                if isinstance(data_item, dict) and 'error' in data_item:
                    error = data_item.get('error')
                    await process.log(f"NASA POWER get data error: {error}")
                    continue
                clean_data.append(data_item)
            item['data'] = clean_data

        final_results.append(item)

    enriched_locations: List[Dict[str, Any]] = []
    for loc_idx, location in enumerate(locations):
        enriched_record = dict(location)
        enriched_record['nasaPowerProperties'] = final_results[loc_idx]
        enriched_locations.append(enriched_record)

    return enriched_locations

def fetch_nasa_power_data_from_api(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    parameter: str,
    frequency: str = 'daily',
    community: str = 'ag'
) -> Dict[str, Any]:
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
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_zarr_with_xarray(start_date, end_date, latitude, longitude, parameter, frequency)


async def fetch_nasa_power_data_batch_async(
    queries: List[Dict[str, Any]],
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    fetcher = NASAPowerDataFetcher()
    return await fetcher.get_data_from_zarr_batch_async(queries, max_concurrent)


async def fetch_nasa_power_data_from_api_batch_async(
    queries: List[Dict[str, Any]],
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    fetcher = NASAPowerDataFetcher()
    return await fetcher.get_data_from_api_batch_async(queries, max_concurrent)


def fetch_nasa_power_data_from_api_batch_multiprocessing(
    queries: List[Dict[str, Any]],
    max_processes: int = 5
) -> List[Dict[str, Any]]:
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_api_batch_multiprocessing(queries, max_processes)


def fetch_nasa_power_data_from_zarr_batch_multiprocessing(
    queries: List[Dict[str, Any]],
    max_processes: int = 5
) -> List[Dict[str, Any]]:
    fetcher = NASAPowerDataFetcher()
    return fetcher.get_data_from_zarr_batch_multiprocessing(queries, max_processes)

