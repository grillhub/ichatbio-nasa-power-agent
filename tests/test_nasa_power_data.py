"""
Tests for NASA POWER data fetching module
"""

import pytest
import json
import os
import time
from datetime import datetime, timedelta
from src.nasa_power_data import (
    NASAPowerDataFetcher, 
    fetch_nasa_power_data_from_zarr_with_xarray,
    fetch_nasa_power_data_from_api,
    enrich_locations_with_nasa_data,
    COMMON_PARAMETERS
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMING_REPORT = []

def add_timing_report(test_name, duration, details=None):
    """Add timing information to the report"""
    report_entry = {
        "test_name": test_name,
        "duration_seconds": round(duration, 3),
        "timestamp": datetime.now().isoformat(),
    }
    if details:
        report_entry["details"] = details
    TIMING_REPORT.append(report_entry)

def save_timing_report(report_name='report.json'):
    """Save timing report to JSON file"""
    report_file = os.path.join(OUTPUT_DIR, report_name)
    with open(report_file, 'w') as f:
        json.dump({
            "total_tests": len(TIMING_REPORT),
            "total_duration_seconds": round(sum(r["duration_seconds"] for r in TIMING_REPORT), 3),
            "tests": TIMING_REPORT
        }, f, indent=2)
    print(f"\nTiming report saved to {report_file}")


class TestNASAPowerData:
    """Test the NASAPowerData class"""
    
    def test_init(self):
        """Test initialization of data fetcher"""
        fetcher = NASAPowerDataFetcher()
        assert fetcher.s3 is not None
        assert fetcher._cache == {}
    
    def test_get_parameter_info(self):
        """Test getting parameter information from API"""
        fetcher = NASAPowerDataFetcher()
        params = fetcher.get_parameter_info()
        
        assert isinstance(params, dict)
        assert 'T2M' in params
        assert 'RH2M' in params
        
        assert isinstance(params['T2M'], str)
        assert isinstance(params['RH2M'], str)
        
        assert len(params['T2M']) > 0
        assert len(params['RH2M']) > 0
    
    def test_invalid_frequency(self):
        """Test that invalid frequency raises ValueError"""
        fetcher = NASAPowerDataFetcher()
        
        queries = [{
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "parameter": "T2M",
            "frequency": "invalid"
        }]
        
        results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
        assert len(results) == 1
        assert 'error' in results[0]
        assert "Invalid frequency" in results[0]['error']
    
    def test_invalid_date_format(self):
        """Test that invalid date format raises ValueError"""
        fetcher = NASAPowerDataFetcher()
        
        queries = [{
            "start_date": "01/01/2024",
            "end_date": "01/31/2024",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "parameter": "T2M",
            "frequency": "daily"
        }]
        
        results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
        assert len(results) == 1
        assert 'error' in results[0]
        assert "Dates must be in YYYY-MM-DD format" in results[0]['error']
    
    def test_invalid_date_range(self):
        """Test that end_date before start_date raises ValueError"""
        fetcher = NASAPowerDataFetcher()
        
        queries = [{
            "start_date": "2024-01-31",
            "end_date": "2024-01-01",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "parameter": "T2M",
            "frequency": "daily"
        }]
        
        results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
        assert len(results) == 1
        assert 'error' in results[0]
        assert "start_date must be before or equal to end_date" in results[0]['error']
    
    def test_invalid_latitude(self):
        """Test that invalid latitude raises ValueError"""
        fetcher = NASAPowerDataFetcher()
        
        queries = [{
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "latitude": 95.0,
            "longitude": -74.0060,
            "parameter": "T2M",
            "frequency": "daily"
        }]
        
        results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
        assert len(results) == 1
        assert 'error' in results[0]
        assert "Latitude must be between -90 and 90" in results[0]['error']
    
    def test_invalid_longitude(self):
        """Test that invalid longitude raises ValueError"""
        fetcher = NASAPowerDataFetcher()
        
        queries = [{
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "latitude": 40.7128,
            "longitude": -200.0,
            "parameter": "T2M",
            "frequency": "daily"
        }]
        
        results = fetcher.get_data_from_zarr_batch_multiprocessing(queries)
        assert len(results) == 1
        assert 'error' in results[0]
        assert "Longitude must be between -180 and 180" in results[0]['error']
    
    def test_enrich_locations_with_valid_data(self):
        """Test enrichment with valid location records"""
        locations = [
            {
                "eventDate": "2023-04-02T10:39",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            },
            {
                "eventDate": "2019-09-09T10:00",
                "decimalLatitude": -21.524045,
                "decimalLongitude": -66.896778
            }
        ]
        
        start_time = time.time()
        result = enrich_locations_with_nasa_data(
            locations=locations,
            parameters=['T2M'],
            date_range_days=1,
            frequency='daily'
        )
        duration = time.time() - start_time
        
        add_timing_report(
            "test_enrich_locations_with_valid_data",
            duration,
            {"locations": len(locations), "parameters": ["T2M"], "frequency": "daily"}
        )
        
        # Verify structure
        assert len(result) == 2
        assert all('nasaPowerProperties' in rec for rec in result)
        
        # Each original field should still be present
        for i, rec in enumerate(result):
            assert rec['eventDate'] == locations[i]['eventDate']
            assert rec['decimalLatitude'] == locations[i]['decimalLatitude']
            assert rec['decimalLongitude'] == locations[i]['decimalLongitude']
    
    def test_enrich_locations_with_null_values(self):
        """Test that records with null values are handled gracefully"""
        locations = [
            {
                "eventDate": "2023-04-02T10:39",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            },
            {
                "eventDate": None,
                "decimalLatitude": None,
                "decimalLongitude": None
            },
            {
                "eventDate": "2019-09-09T10:00",
                "decimalLatitude": -21.524045,
                "decimalLongitude": None
            }
        ]
        
        start_time = time.time()
        result = enrich_locations_with_nasa_data(
            locations=locations,
            parameters=['T2M'],
            date_range_days=1,
            frequency='daily'
        )
        duration = time.time() - start_time
        
        add_timing_report(
            "test_enrich_locations_with_null_values",
            duration,
            {"locations": len(locations), "valid_locations": 1, "null_locations": 2}
        )
        
        # Verify all records are returned
        assert len(result) == 3
        
        # Records with null values should have nasaPowerProperties set to None
        assert result[1]['nasaPowerProperties'] is None
        assert result[2]['nasaPowerProperties'] is None
    
    def test_enrich_locations_with_different_date_formats(self):
        """Test handling of different ISO date formats"""
        locations = [
            {
                "eventDate": "2023-04-02T10:39",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            },
            {
                "eventDate": "2023-04-02T10:39:30Z",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            },
            {
                "eventDate": "2023-04-02",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            }
        ]
        
        start_time = time.time()
        result = enrich_locations_with_nasa_data(
            locations=locations,
            parameters=['T2M'],
            date_range_days=1,
            frequency='daily'
        )
        duration = time.time() - start_time
        
        add_timing_report(
            "test_enrich_locations_with_different_date_formats",
            duration,
            {"locations": len(locations), "date_formats": ["ISO_with_time", "ISO_with_timezone", "date_only"]}
        )
        
        # All should have nasaPowerProperties
        assert len(result) == 3
        assert all('nasaPowerProperties' in rec for rec in result)
    
    def test_enrich_locations_with_invalid_date_format(self):
        """Test that invalid date formats are handled gracefully"""
        locations = [
            {
                "eventDate": "invalid-date",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            }
        ]
        
        start_time = time.time()
        result = enrich_locations_with_nasa_data(
            locations=locations,
            parameters=['T2M'],
            date_range_days=1,
            frequency='daily'
        )
        duration = time.time() - start_time
        
        add_timing_report(
            "test_enrich_locations_with_invalid_date_format",
            duration,
            {"locations": len(locations), "invalid_dates": 1}
        )
        
        # Should return record with nasaPowerProperties set to None
        assert len(result) == 1
        assert result[0]['nasaPowerProperties'] is None
    
    def test_enrich_locations_with_multiple_parameters(self):
        """Test enrichment with multiple NASA POWER parameters"""
        locations = [
            {
                "eventDate": "2023-04-02T10:39",
                "decimalLatitude": -50.948586,
                "decimalLongitude": -72.712978
            }
        ]
        
        start_time = time.time()
        result = enrich_locations_with_nasa_data(
            locations=locations,
            parameters=['T2M', 'RH2M'],
            date_range_days=1,
            frequency='daily'
        )
        duration = time.time() - start_time
        
        add_timing_report(
            "test_enrich_locations_with_multiple_parameters",
            duration,
            {"locations": 1, "parameters": ["T2M", "RH2M"], "parameter_count": 2}
        )
        
        # Verify structure
        assert len(result) == 1
        assert 'nasaPowerProperties' in result[0]
        
        # Should have data for both parameters (or error info if not available)
        nasa_props = result[0]['nasaPowerProperties']
        if nasa_props is not None:
            assert len(nasa_props) == 2
    
    def test_enrich_locations_empty_list(self):
        """Test enrichment with empty location list"""
        result = enrich_locations_with_nasa_data(
            locations=[],
            parameters=['T2M'],
            date_range_days=1,
            frequency='daily'
        )
        
        assert result == []
    
    def test_zarr_multiprocessing_vs_api_comparison(self):
        """Test that Zarr multiprocessing and API return similar results for 5 locations"""

        test_locations = [
            {
                "latitude": 51.5074,  # London, UK
                "longitude": -0.1278,
                "date": "2024-03-20"
            },
            {
                "latitude": 35.6762,  # Tokyo, Japan
                "longitude": 139.6503,
                "date": "2024-06-15"
            },
            {
                "latitude": -33.8688,  # Sydney, Australia
                "longitude": 151.2093,
                "date": "2024-12-25"
            },
            {
                "latitude": -22.9068,  # Rio de Janeiro, Brazil
                "longitude": -43.1729,
                "date": "2024-08-10"
            },
            {
                "latitude": 28.6139,  # New Delhi, India
                "longitude": 77.2090,
                "date": "2024-05-01"
            }
        ]
        
        parameter = 'T2M'
        frequency = 'daily'
        
        queries = []
        for loc in test_locations:
            queries.append({
                'start_date': loc['date'],
                'end_date': loc['date'],
                'latitude': loc['latitude'],
                'longitude': loc['longitude'],
                'parameter': parameter,
                'frequency': frequency
            })
        
        fetcher = NASAPowerDataFetcher()
        
        start_time = time.time()
        zarr_results = fetcher.get_data_from_zarr_batch_multiprocessing(queries, max_processes=5)
        zarr_duration = time.time() - start_time
        
        api_start_time = time.time()
        api_results = []
        for query in queries:
            try:
                result = fetcher.get_data_from_api(
                    start_date=query['start_date'],
                    end_date=query['end_date'],
                    latitude=query['latitude'],
                    longitude=query['longitude'],
                    parameter=query['parameter'],
                    frequency=query['frequency']
                )
                api_results.append(result)
            except Exception as e:
                api_results.append({'error': str(e)})
        api_duration = time.time() - api_start_time
        total_duration = time.time() - start_time
        
        zarr_success = sum(1 for r in zarr_results if 'error' not in r)
        api_success = sum(1 for r in api_results if 'error' not in r)
        
        add_timing_report(
            "test_zarr_multiprocessing_vs_api_comparison",
            total_duration,
            {
                "locations": len(test_locations),
                "zarr_duration_seconds": round(zarr_duration, 3),
                "api_duration_seconds": round(api_duration, 3),
                "zarr_success_count": zarr_success,
                "api_success_count": api_success,
                "parameter": parameter,
                "frequency": frequency
            }
        )
        
        assert len(zarr_results) == len(queries), "Zarr should return results for all queries"
        assert len(api_results) == len(queries), "API should return results for all queries"
        
        assert zarr_success > 0, f"At least one Zarr query should succeed (got {zarr_success})"
        assert api_success > 0, f"At least one API query should succeed (got {api_success})"
        
        for i, (loc, zarr_result, api_result) in enumerate(zip(test_locations, zarr_results, api_results)):
            zarr_has_error = 'error' in zarr_result
            api_has_error = 'error' in api_result
            
            if not zarr_has_error:
                assert 'parameter' in zarr_result, f"Location {i+1}: Zarr result missing 'parameter'"
                assert 'frequency' in zarr_result, f"Location {i+1}: Zarr result missing 'frequency'"
                assert 'latitude' in zarr_result, f"Location {i+1}: Zarr result missing 'latitude'"
                assert 'longitude' in zarr_result, f"Location {i+1}: Zarr result missing 'longitude'"
                assert 'data' in zarr_result, f"Location {i+1}: Zarr result missing 'data'"
                assert zarr_result['parameter'] == parameter, f"Location {i+1}: Zarr parameter mismatch"
                assert zarr_result['frequency'] == frequency, f"Location {i+1}: Zarr frequency mismatch"
                assert isinstance(zarr_result['data'], list), f"Location {i+1}: Zarr data should be a list"
                assert len(zarr_result['data']) > 0, f"Location {i+1}: Zarr data should not be empty"
            
            if not api_has_error:
                assert 'parameter' in api_result, f"Location {i+1}: API result missing 'parameter'"
                assert 'frequency' in api_result, f"Location {i+1}: API result missing 'frequency'"
                assert 'latitude' in api_result, f"Location {i+1}: API result missing 'latitude'"
                assert 'longitude' in api_result, f"Location {i+1}: API result missing 'longitude'"
                assert 'data' in api_result, f"Location {i+1}: API result missing 'data'"
                assert api_result['parameter'] == parameter, f"Location {i+1}: API parameter mismatch"
                assert api_result['frequency'] == frequency, f"Location {i+1}: API frequency mismatch"
                assert isinstance(api_result['data'], list), f"Location {i+1}: API data should be a list"
                assert len(api_result['data']) > 0, f"Location {i+1}: API data should not be empty"
            
            if not zarr_has_error and not api_has_error:
                zarr_data = zarr_result['data']
                api_data = api_result['data']
                
                assert len(zarr_data) > 0, f"Location {i+1}: Zarr should have data points"
                assert len(api_data) > 0, f"Location {i+1}: API should have data points"
                
                zarr_point = zarr_data[0]
                api_point = api_data[0]
                
                assert 'date' in zarr_point, f"Location {i+1}: Zarr data point missing 'date'"
                assert 'value' in zarr_point, f"Location {i+1}: Zarr data point missing 'value'"
                assert 'date' in api_point, f"Location {i+1}: API data point missing 'date'"
                assert 'value' in api_point, f"Location {i+1}: API data point missing 'value'"
                
                assert zarr_point['date'] == api_point['date'], \
                    f"Location {i+1}: Date mismatch - Zarr: {zarr_point['date']}, API: {api_point['date']}"
                
                # Both should have valid coordinates (may differ due to grid resolution)
                assert isinstance(zarr_result['latitude'], (int, float)), \
                    f"Location {i+1}: Zarr latitude should be numeric"
                assert isinstance(zarr_result['longitude'], (int, float)), \
                    f"Location {i+1}: Zarr longitude should be numeric"
                assert isinstance(api_result['latitude'], (int, float)), \
                    f"Location {i+1}: API latitude should be numeric"
                assert isinstance(api_result['longitude'], (int, float)), \
                    f"Location {i+1}: API longitude should be numeric"
                
                # Compare values - round to 2 decimal places for comparison
                zarr_value = zarr_point.get('value')
                api_value = api_point.get('value')
                
                if zarr_value is None and api_value is None:
                    pass
                elif zarr_value is None or api_value is None:
                    assert False, \
                        f"Location {i+1}: One value is None - Zarr: {zarr_value}, API: {api_value}"
                else:
                    zarr_rounded = round(zarr_value, 2)
                    api_rounded = round(api_value, 2)
                    assert zarr_rounded == api_rounded, \
                        f"Location {i+1}: Value mismatch - Zarr: {zarr_value} (rounded: {zarr_rounded}), API: {api_value} (rounded: {api_rounded})"
