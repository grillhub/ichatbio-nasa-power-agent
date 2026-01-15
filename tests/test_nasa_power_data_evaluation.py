"""
Evaluation tests for NASA POWER data - comparing API vs Zarr performance and results
"""

import pytest
import json
import os
import time
import asyncio
import statistics
from datetime import datetime, timedelta
from src.nasa_power_data import (
    NASAPowerDataFetcher, 
    enrich_locations_with_nasa_data,
    fetch_nasa_power_data_batch_async,
    fetch_nasa_power_data_from_api_batch_async,
    COMMON_PARAMETERS
)

# Output directory for all test outputs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TestZarrBatch:
    """Test batch processing for Zarr data fetching - async, sync, and multiprocessing"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_filename,output_filename,max_concurrent,test_name", [
        ("list_of_locations_1000_full.json", "output_zarr_batch_1000locations_10concurrent.json", 10, "test_zarr_batch_1000l_10c"),
        ("list_of_locations_1000_full.json", "output_zarr_batch_1000locations_100concurrent.json", 100, "test_zarr_batch_1000l_100c"),
        ("list_of_locations_1000_full.json", "output_zarr_batch_1000locations_1000concurrent.json", 1000, "test_zarr_batch_1000l_1000c"),
        ("list_of_locations_20000_full.json", "output_zarr_batch_20000locations_100concurrent.json", 100, "test_zarr_batch_20000l_100c"),
    ])
    async def test_zarr_batch(self, input_filename, output_filename, max_concurrent, test_name):
        """Test batch fetching with async, sync, and multiprocessing"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(test_dir, "resources", input_filename)
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # Read input data
        with open(input_file, 'r') as f:
            locations = json.load(f)
        
        # Filter to valid locations only
        valid_locations = [
            loc for loc in locations
            if loc.get('eventDate') and loc.get('decimalLatitude') and loc.get('decimalLongitude')
        ]
        
        print(f"\n{'='*80}")
        print(f"ASYNC BATCH TEST: {test_name}")
        print(f"   Input: {input_filename}")
        print(f"   Total locations: {len(locations)}")
        print(f"   Valid locations: {len(valid_locations)}")
        print(f"   Max concurrent: {max_concurrent}")
        print(f"{'='*80}")
        
        # Build queries from valid locations
        parameter = 'T2M'
        queries = []
        
        for loc in valid_locations:
            event_date = loc['eventDate']
            # Parse date
            if 'T' in event_date:
                date_str = event_date.split('T')[0]
            else:
                date_str = event_date
            
            queries.append({
                'start_date': date_str,
                'end_date': date_str,
                'latitude': float(loc['decimalLatitude']),
                'longitude': float(loc['decimalLongitude']),
                'parameter': parameter,
                'frequency': 'daily'
            })
        
        print(f"\n  Built {len(queries)} queries")
        
        # Run async batch fetch with timing
        print(f"\n  Running async batch fetch with {max_concurrent} concurrent requests...")
        
        fetcher = NASAPowerDataFetcher()
        
        start_time = time.time()
        results = await fetcher.get_data_from_zarr_batch_async(queries, max_concurrent=max_concurrent, include_timing=True)
        async_duration = time.time() - start_time
        
        # Analyze results - separate success vs error timing
        success_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        success_count = len(success_results)
        error_count = len(error_results)
        
        # Calculate timing stats for successful requests only
        success_times = [r.get('_timing_ms', 0) for r in success_results]
        error_times = [r.get('_timing_ms', 0) for r in error_results]
        
        if success_times:
            success_avg_ms = statistics.mean(success_times)
            success_std_ms = statistics.stdev(success_times) if len(success_times) > 1 else 0
            success_min_ms = min(success_times)
            success_max_ms = max(success_times)
        else:
            success_avg_ms = success_std_ms = success_min_ms = success_max_ms = 0
        
        if error_times:
            error_avg_ms = statistics.mean(error_times)
        else:
            error_avg_ms = 0
        
        print(f"\n  Completed in {async_duration:.2f}s")
        print(f"     Success: {success_count}/{len(queries)} ({success_count/len(queries)*100:.1f}%)")
        print(f"     Errors: {error_count}/{len(queries)}")
        print(f"     Throughput: {len(queries)/async_duration:.1f} queries/sec")
        if success_times:
            print(f"     Successful request time: {success_avg_ms:.1f}ms avg ± {success_std_ms:.1f}ms std")
            print(f"        Range: {success_min_ms:.1f}ms - {success_max_ms:.1f}ms")
        if error_times:
            print(f"     Failed request time: {error_avg_ms:.1f}ms avg (fast failures indicate rate limiting)")
        
        # Compare with synchronous execution (sample) - track individual times
        sync_sample_size = min(20, len(queries))
        print(f"\n  Running synchronous comparison ({sync_sample_size} queries)...")
        
        sync_start_time = time.time()
        sync_results = []
        sync_individual_times = []  # Track individual query times
        
        for i, query in enumerate(queries[:sync_sample_size]):
            query_start = time.time()
            try:
                result = fetcher.get_data_from_zarr_with_xarray(
                    start_date=query['start_date'],
                    end_date=query['end_date'],
                    latitude=query['latitude'],
                    longitude=query['longitude'],
                    parameter=query['parameter'],
                    frequency=query['frequency']
                )
                sync_results.append(result)
            except Exception as e:
                sync_results.append({'error': str(e)})
            query_duration = time.time() - query_start
            sync_individual_times.append(query_duration * 1000)  # Convert to ms
        
        sync_duration = time.time() - sync_start_time
        
        sync_success_count = sum(1 for r in sync_results if 'error' not in r)
        
        # Calculate statistics for sync times
        sync_avg_ms = statistics.mean(sync_individual_times)
        sync_std_ms = statistics.stdev(sync_individual_times) if len(sync_individual_times) > 1 else 0
        sync_min_ms = min(sync_individual_times)
        sync_max_ms = max(sync_individual_times)
        sync_median_ms = statistics.median(sync_individual_times)
        
        print(f"     Sync completed in {sync_duration:.2f}s")
        print(f"     Sync success: {sync_success_count}/{sync_sample_size}")
        print(f"     Sync avg per query: {sync_avg_ms:.1f}ms (std: {sync_std_ms:.1f}ms)")
        
        # Run multiprocessing batch fetch
        # Use same concurrency as async, but cap at 50 processes (system limit)
        mp_processes = min(max_concurrent, 50, len(queries))
        print(f"\n  Running multiprocessing batch fetch with {mp_processes} processes...")
        
        mp_start_time = time.time()
        mp_results = fetcher.get_data_from_zarr_batch_multiprocessing(queries, max_processes=mp_processes, include_timing=True)
        mp_duration = time.time() - mp_start_time
        
        mp_success_results = [r for r in mp_results if 'error' not in r]
        mp_success_count = len(mp_success_results)
        mp_error_count = len(queries) - mp_success_count
        
        mp_success_times = [r.get('_timing_ms', 0) for r in mp_success_results]
        if mp_success_times:
            mp_avg_ms = statistics.mean(mp_success_times)
            mp_std_ms = statistics.stdev(mp_success_times) if len(mp_success_times) > 1 else 0
        else:
            mp_avg_ms = mp_std_ms = 0
        
        print(f"     Multiprocessing completed in {mp_duration:.2f}s")
        print(f"     Success: {mp_success_count}/{len(queries)} ({mp_success_count/len(queries)*100:.1f}%)")
        print(f"     Throughput: {len(queries)/mp_duration:.1f} queries/sec")
        if mp_success_times:
            print(f"     Request time: {mp_avg_ms:.1f}ms avg ± {mp_std_ms:.1f}ms std")
        
        # Calculate speedups
        async_time_per_query = async_duration / len(queries)
        sync_time_per_query = sync_duration / sync_sample_size
        mp_time_per_query = mp_duration / len(queries)
        
        async_vs_sync_speedup = sync_time_per_query / async_time_per_query if async_time_per_query > 0 else 0
        mp_vs_sync_speedup = sync_time_per_query / mp_time_per_query if mp_time_per_query > 0 else 0
        async_vs_mp_speedup = mp_duration / async_duration if async_duration > 0 else 0
        
        # Build summary
        summary = {
            'test_info': {
                'test_name': test_name,
                'input_file': input_filename,
                'total_locations': len(locations),
                'valid_locations': len(valid_locations),
                'queries_executed': len(queries),
                'max_concurrent': max_concurrent,
                'mp_processes': mp_processes,
                'parameter': parameter,
                'data_source': 'Zarr'
            },
            'async_performance': {
                'total_duration_seconds': round(async_duration, 3),
                'success_count': success_count,
                'error_count': error_count,
                'success_rate_percent': round(success_count / len(queries) * 100, 2),
                'avg_time_per_query_ms': round(async_duration / len(queries) * 1000, 2),
                'throughput_queries_per_sec': round(len(queries) / async_duration, 2),
                'successful_request_timing': {
                    'avg_ms': round(success_avg_ms, 2),
                    'std_ms': round(success_std_ms, 2),
                    'min_ms': round(success_min_ms, 2),
                    'max_ms': round(success_max_ms, 2)
                } if success_times else None,
                'failed_request_avg_ms': round(error_avg_ms, 2) if error_times else None
            },
            'sync_comparison': {
                'sample_size': sync_sample_size,
                'total_duration_seconds': round(sync_duration, 3),
                'success_count': sync_success_count,
                'avg_time_per_query_ms': round(sync_avg_ms, 2),
                'std_time_per_query_ms': round(sync_std_ms, 2),
                'min_time_per_query_ms': round(sync_min_ms, 2),
                'max_time_per_query_ms': round(sync_max_ms, 2),
                'median_time_per_query_ms': round(sync_median_ms, 2),
                'throughput_queries_per_sec': round(sync_sample_size / sync_duration, 2) if sync_duration > 0 else 0
            },
            'multiprocessing_performance': {
                'processes': mp_processes,
                'total_duration_seconds': round(mp_duration, 3),
                'success_count': mp_success_count,
                'error_count': mp_error_count,
                'success_rate_percent': round(mp_success_count / len(queries) * 100, 2),
                'avg_time_per_query_ms': round(mp_duration / len(queries) * 1000, 2),
                'throughput_queries_per_sec': round(len(queries) / mp_duration, 2),
                'request_timing': {
                    'avg_ms': round(mp_avg_ms, 2),
                    'std_ms': round(mp_std_ms, 2)
                } if mp_success_times else None
            },
            'speedup_comparison': {
                'async_vs_sync_speedup': round(async_vs_sync_speedup, 2),
                'mp_vs_sync_speedup': round(mp_vs_sync_speedup, 2),
                'async_vs_mp_speedup': round(async_vs_mp_speedup, 2),
                'fastest_method': 'async' if async_duration < mp_duration else 'multiprocessing',
                'estimated_sync_time_for_all': round(sync_time_per_query * len(queries), 2)
            }
        }
        
        # Collect sample results
        sample_results = []
        for i, (query, result) in enumerate(zip(queries[:10], results[:10])):
            sample_entry = {
                'index': i,
                'query': query,
                'status': 'success' if 'error' not in result else 'error'
            }
            if 'error' in result:
                sample_entry['error'] = result['error']
            else:
                sample_entry['actual_latitude'] = result.get('latitude')
                sample_entry['actual_longitude'] = result.get('longitude')
                sample_entry['data_points'] = len(result.get('data', []))
                if result.get('data'):
                    sample_entry['first_value'] = result['data'][0]
            sample_results.append(sample_entry)
        
        # Write output
        output_data = {
            'summary': summary,
            'sample_results': sample_results,
            'all_results_count': len(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nReport saved to: {output_file}")
        print(f"{'='*80}\n")
        
        # Add to timing report
        # add_timing_report(test_name, async_duration + mp_duration, summary)
        
        # Assertions
        assert len(results) == len(queries), "Should return same number of results as queries"
        assert len(mp_results) == len(queries), "Multiprocessing should return same number of results"
        # At least one method should have some success (may hit rate limits)
        total_successes = success_count + mp_success_count
        assert total_successes > 0, f"At least one method should succeed (async={success_count}, mp={mp_success_count})"

class TestApiBatch:
    """Test batch processing for API data fetching - async, sync, and multiprocessing"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_filename,output_filename,max_concurrent,test_name", [
        ("list_of_locations_1000_full.json", "output_api_batch_10concurrent.json", 10, "test_api_batch_1000l_10c"),
        ("list_of_locations_1000_full.json", "output_api_batch_100concurrent.json", 100, "test_api_batch_1000l_100c"),
        ("list_of_locations_1000_full.json", "output_api_batch_1000concurrent.json", 1000, "test_api_batch_1000l_100c"),
        ("list_of_locations_20000_full.json", "output_api_batch_20000locations_100concurrent.json", 100, "test_api_batch_20000l_100c"),
    ])
    async def test_api_batch(self, input_filename, output_filename, max_concurrent, test_name):
        """Test batch API fetching with async, sync, and multiprocessing"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(test_dir, "resources", input_filename)
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # Read input data
        with open(input_file, 'r') as f:
            locations = json.load(f)
        
        # Filter to valid locations only (no limit - use all valid locations)
        valid_locations = [
            loc for loc in locations
            if loc.get('eventDate') and loc.get('decimalLatitude') and loc.get('decimalLongitude')
        ]
        
        print(f"\n{'='*80}")
        print(f"API ASYNC BATCH TEST: {test_name}")
        print(f"   Input: {input_filename}")
        print(f"   Total locations: {len(locations)}")
        print(f"   Valid locations: {len(valid_locations)}")
        print(f"   Max concurrent: {max_concurrent}")
        print(f"{'='*80}")
        
        # Build queries from valid locations
        parameter = 'T2M'
        queries = []
        
        for loc in valid_locations:
            event_date = loc['eventDate']
            # Parse date
            if 'T' in event_date:
                date_str = event_date.split('T')[0]
            else:
                date_str = event_date
            
            queries.append({
                'start_date': date_str,
                'end_date': date_str,
                'latitude': float(loc['decimalLatitude']),
                'longitude': float(loc['decimalLongitude']),
                'parameter': parameter,
                'frequency': 'daily'
            })
        
        print(f"\n  Built {len(queries)} queries")
        
        # Run async batch fetch with timing
        print(f"\n  Running async API batch fetch with {max_concurrent} concurrent requests...")
        
        fetcher = NASAPowerDataFetcher()
        
        start_time = time.time()
        results = await fetcher.get_data_from_api_batch_async(queries, max_concurrent=max_concurrent, include_timing=True)
        async_duration = time.time() - start_time
        
        # Analyze results - separate success vs error timing
        success_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        success_count = len(success_results)
        error_count = len(error_results)
        
        # Calculate timing stats for successful requests only
        success_times = [r.get('_timing_ms', 0) for r in success_results]
        error_times = [r.get('_timing_ms', 0) for r in error_results]
        
        if success_times:
            success_avg_ms = statistics.mean(success_times)
            success_std_ms = statistics.stdev(success_times) if len(success_times) > 1 else 0
            success_min_ms = min(success_times)
            success_max_ms = max(success_times)
        else:
            success_avg_ms = success_std_ms = success_min_ms = success_max_ms = 0
        
        if error_times:
            error_avg_ms = statistics.mean(error_times)
        else:
            error_avg_ms = 0
        
        # Collect error types
        error_types = {}
        for r in error_results:
            error_msg = r.get('error', 'Unknown')[:50]  # First 50 chars
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        print(f"\n  Completed in {async_duration:.2f}s")
        print(f"     Success: {success_count}/{len(queries)} ({success_count/len(queries)*100:.1f}%)")
        print(f"     Errors: {error_count}/{len(queries)}")
        print(f"     Throughput: {len(queries)/async_duration:.1f} queries/sec")
        if success_times:
            print(f"     Successful request time: {success_avg_ms:.1f}ms avg ± {success_std_ms:.1f}ms std")
            print(f"        Range: {success_min_ms:.1f}ms - {success_max_ms:.1f}ms")
        if error_times:
            print(f"     Failed request time: {error_avg_ms:.1f}ms avg (fast failures = rate limiting)")
        if error_types:
            print(f"     Error types:")
            for err, count in sorted(error_types.items(), key=lambda x: -x[1])[:3]:
                print(f"        - {err}: {count} occurrences")
        
        # Compare with synchronous execution (sample) - track individual times
        sync_sample_size = min(20, len(queries))
        print(f"\n  Running synchronous API comparison ({sync_sample_size} queries)...")
        
        sync_start_time = time.time()
        sync_results = []
        sync_individual_times = []  # Track individual query times
        
        for i, query in enumerate(queries[:sync_sample_size]):
            query_start = time.time()
            try:
                result = fetcher.get_data_from_api(
                    start_date=query['start_date'],
                    end_date=query['end_date'],
                    latitude=query['latitude'],
                    longitude=query['longitude'],
                    parameter=query['parameter'],
                    frequency=query['frequency']
                )
                sync_results.append(result)
            except Exception as e:
                sync_results.append({'error': str(e)})
            query_duration = time.time() - query_start
            sync_individual_times.append(query_duration * 1000)  # Convert to ms
        
        sync_duration = time.time() - sync_start_time
        
        sync_success_count = sum(1 for r in sync_results if 'error' not in r)
        
        # Calculate statistics for sync times
        sync_avg_ms = statistics.mean(sync_individual_times) if sync_individual_times else 0
        sync_std_ms = statistics.stdev(sync_individual_times) if len(sync_individual_times) > 1 else 0
        sync_min_ms = min(sync_individual_times) if sync_individual_times else 0
        sync_max_ms = max(sync_individual_times) if sync_individual_times else 0
        sync_median_ms = statistics.median(sync_individual_times) if sync_individual_times else 0
        
        print(f"     Sync completed in {sync_duration:.2f}s")
        print(f"     Sync success: {sync_success_count}/{sync_sample_size}")
        print(f"     Sync avg per query: {sync_avg_ms:.1f}ms (std: {sync_std_ms:.1f}ms)")
        
        # Run multiprocessing batch fetch
        # Use same concurrency as async, but cap at 50 processes (system limit)
        mp_processes = min(max_concurrent, 50, len(queries))
        print(f"\n  Running multiprocessing API batch fetch with {mp_processes} processes...")
        
        mp_start_time = time.time()
        mp_results = fetcher.get_data_from_api_batch_multiprocessing(queries, max_processes=mp_processes, include_timing=True)
        mp_duration = time.time() - mp_start_time
        
        mp_success_results = [r for r in mp_results if 'error' not in r]
        mp_success_count = len(mp_success_results)
        mp_error_count = len(queries) - mp_success_count
        
        mp_success_times = [r.get('_timing_ms', 0) for r in mp_success_results]
        if mp_success_times:
            mp_avg_ms = statistics.mean(mp_success_times)
            mp_std_ms = statistics.stdev(mp_success_times) if len(mp_success_times) > 1 else 0
        else:
            mp_avg_ms = mp_std_ms = 0
        
        print(f"     Multiprocessing completed in {mp_duration:.2f}s")
        print(f"     Success: {mp_success_count}/{len(queries)} ({mp_success_count/len(queries)*100:.1f}%)")
        print(f"     Throughput: {len(queries)/mp_duration:.1f} queries/sec")
        if mp_success_times:
            print(f"     Request time: {mp_avg_ms:.1f}ms avg ± {mp_std_ms:.1f}ms std")
        
        # Calculate speedups
        async_time_per_query = async_duration / len(queries)
        sync_time_per_query = sync_duration / sync_sample_size if sync_sample_size > 0 else 0
        mp_time_per_query = mp_duration / len(queries)
        
        async_vs_sync_speedup = sync_time_per_query / async_time_per_query if async_time_per_query > 0 else 0
        mp_vs_sync_speedup = sync_time_per_query / mp_time_per_query if mp_time_per_query > 0 else 0
        async_vs_mp_speedup = mp_duration / async_duration if async_duration > 0 else 0
        
        # Build summary
        summary = {
            'test_info': {
                'test_name': test_name,
                'input_file': input_filename,
                'total_locations': len(locations),
                'valid_locations': len(valid_locations),
                'queries_executed': len(queries),
                'max_concurrent': max_concurrent,
                'mp_processes': mp_processes,
                'parameter': parameter,
                'data_source': 'API'
            },
            'async_performance': {
                'total_duration_seconds': round(async_duration, 3),
                'success_count': success_count,
                'error_count': error_count,
                'success_rate_percent': round(success_count / len(queries) * 100, 2),
                'avg_time_per_query_ms': round(async_duration / len(queries) * 1000, 2),
                'throughput_queries_per_sec': round(len(queries) / async_duration, 2),
                'successful_request_timing': {
                    'avg_ms': round(success_avg_ms, 2),
                    'std_ms': round(success_std_ms, 2),
                    'min_ms': round(success_min_ms, 2),
                    'max_ms': round(success_max_ms, 2)
                } if success_times else None,
                'failed_request_avg_ms': round(error_avg_ms, 2) if error_times else None,
                'error_types': error_types if error_types else None
            },
            'sync_comparison': {
                'sample_size': sync_sample_size,
                'total_duration_seconds': round(sync_duration, 3),
                'success_count': sync_success_count,
                'avg_time_per_query_ms': round(sync_avg_ms, 2),
                'std_time_per_query_ms': round(sync_std_ms, 2),
                'min_time_per_query_ms': round(sync_min_ms, 2),
                'max_time_per_query_ms': round(sync_max_ms, 2),
                'median_time_per_query_ms': round(sync_median_ms, 2),
                'throughput_queries_per_sec': round(sync_sample_size / sync_duration, 2) if sync_duration > 0 else 0
            },
            'multiprocessing_performance': {
                'processes': mp_processes,
                'total_duration_seconds': round(mp_duration, 3),
                'success_count': mp_success_count,
                'error_count': mp_error_count,
                'success_rate_percent': round(mp_success_count / len(queries) * 100, 2),
                'avg_time_per_query_ms': round(mp_duration / len(queries) * 1000, 2),
                'throughput_queries_per_sec': round(len(queries) / mp_duration, 2),
                'request_timing': {
                    'avg_ms': round(mp_avg_ms, 2),
                    'std_ms': round(mp_std_ms, 2)
                } if mp_success_times else None
            },
            'speedup_comparison': {
                'async_vs_sync_speedup': round(async_vs_sync_speedup, 2),
                'mp_vs_sync_speedup': round(mp_vs_sync_speedup, 2),
                'async_vs_mp_speedup': round(async_vs_mp_speedup, 2),
                'fastest_method': 'async' if async_duration < mp_duration else 'multiprocessing',
                'estimated_sync_time_for_all': round(sync_time_per_query * len(queries), 2)
            }
        }
        
        # Collect sample results
        sample_results = []
        for i, (query, result) in enumerate(zip(queries[:10], results[:10])):
            sample_entry = {
                'index': i,
                'query': query,
                'status': 'success' if 'error' not in result else 'error'
            }
            if 'error' in result:
                sample_entry['error'] = result['error']
            else:
                sample_entry['actual_latitude'] = result.get('latitude')
                sample_entry['actual_longitude'] = result.get('longitude')
                sample_entry['data_points'] = len(result.get('data', []))
                if result.get('data'):
                    sample_entry['first_value'] = result['data'][0]
            sample_results.append(sample_entry)
        
        # Write output
        output_data = {
            'summary': summary,
            'sample_results': sample_results,
            'all_results_count': len(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📄 Report saved to: {output_file}")
        print(f"{'='*80}\n")
        
        # Add to timing report
        # add_timing_report(test_name, async_duration + mp_duration, summary)
        
        # Assertions
        assert len(results) == len(queries), "Should return same number of results as queries"
        assert len(mp_results) == len(queries), "Multiprocessing should return same number of results"
        # At least one method should have some success (API may hit rate limits)
        total_successes = success_count + mp_success_count
        assert total_successes > 0, f"At least one method should succeed (async={success_count}, mp={mp_success_count})"

    @pytest.mark.parametrize("input_filename,output_filename,max_processes,test_name", [
        ("list_of_locations_1000.json", "output_api_batch_mp_10processes.json", 10, "test_api_batch_mp_10"),
        ("list_of_locations_1000.json", "output_api_batch_mp_50processes.json", 50, "test_api_batch_mp_50"),
    ])
    def test_api_batch_multiprocessing_only(self, input_filename, output_filename, max_processes, test_name):
        """Test batch API fetching with multiprocessing only"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(test_dir, "resources", input_filename)
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # Read input data
        with open(input_file, 'r') as f:
            locations = json.load(f)
        
        # Filter to valid locations only
        valid_locations = [
            loc for loc in locations
            if loc.get('eventDate') and loc.get('decimalLatitude') and loc.get('decimalLongitude')
        ]
        
        print(f"\n{'='*80}")
        print(f"API MULTIPROCESSING ONLY TEST: {test_name}")
        print(f"   Input: {input_filename}")
        print(f"   Total locations: {len(locations)}")
        print(f"   Valid locations: {len(valid_locations)}")
        print(f"   Max processes: {max_processes}")
        print(f"{'='*80}")
        
        # Build queries from valid locations
        parameter = 'T2M'
        queries = []
        
        for loc in valid_locations:
            event_date = loc['eventDate']
            # Parse date
            if 'T' in event_date:
                date_str = event_date.split('T')[0]
            else:
                date_str = event_date
            
            queries.append({
                'start_date': date_str,
                'end_date': date_str,
                'latitude': float(loc['decimalLatitude']),
                'longitude': float(loc['decimalLongitude']),
                'parameter': parameter,
                'frequency': 'daily'
            })
        
        print(f"\n  Built {len(queries)} queries")
        
        fetcher = NASAPowerDataFetcher()
        
        # Run multiprocessing batch fetch
        mp_processes = min(max_processes, len(queries))
        print(f"\n  Running multiprocessing API batch fetch with {mp_processes} processes...")
        
        mp_start_time = time.time()
        mp_results = fetcher.get_data_from_api_batch_multiprocessing(queries, max_processes=mp_processes, include_timing=True)
        mp_duration = time.time() - mp_start_time
        
        mp_success_results = [r for r in mp_results if 'error' not in r]
        mp_error_results = [r for r in mp_results if 'error' in r]
        mp_success_count = len(mp_success_results)
        mp_error_count = len(mp_error_results)
        
        mp_success_times = [r.get('_timing_ms', 0) for r in mp_success_results]
        if mp_success_times:
            mp_avg_ms = statistics.mean(mp_success_times)
            mp_std_ms = statistics.stdev(mp_success_times) if len(mp_success_times) > 1 else 0
            mp_min_ms = min(mp_success_times)
            mp_max_ms = max(mp_success_times)
        else:
            mp_avg_ms = mp_std_ms = mp_min_ms = mp_max_ms = 0
        
        # Collect error types
        error_types = {}
        for r in mp_error_results:
            error_msg = r.get('error', 'Unknown')[:50]
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        print(f"\n  Multiprocessing completed in {mp_duration:.2f}s")
        print(f"     Success: {mp_success_count}/{len(queries)} ({mp_success_count/len(queries)*100:.1f}%)")
        print(f"     Errors: {mp_error_count}/{len(queries)}")
        print(f"     Throughput: {len(queries)/mp_duration:.1f} queries/sec")
        if mp_success_times:
            print(f"     Request time: {mp_avg_ms:.1f}ms avg ± {mp_std_ms:.1f}ms std")
            print(f"        Range: {mp_min_ms:.1f}ms - {mp_max_ms:.1f}ms")
        if error_types:
            print(f"     Error types:")
            for err, count in sorted(error_types.items(), key=lambda x: -x[1])[:3]:
                print(f"        - {err}: {count} occurrences")
        
        # Build summary
        summary = {
            'test_info': {
                'test_name': test_name,
                'input_file': input_filename,
                'total_locations': len(locations),
                'valid_locations': len(valid_locations),
                'queries_executed': len(queries),
                'max_processes': max_processes,
                'actual_processes': mp_processes,
                'parameter': parameter,
                'data_source': 'API'
            },
            'multiprocessing_performance': {
                'total_duration_seconds': round(mp_duration, 3),
                'success_count': mp_success_count,
                'error_count': mp_error_count,
                'success_rate_percent': round(mp_success_count / len(queries) * 100, 2),
                'avg_time_per_query_ms': round(mp_duration / len(queries) * 1000, 2),
                'throughput_queries_per_sec': round(len(queries) / mp_duration, 2),
                'request_timing': {
                    'avg_ms': round(mp_avg_ms, 2),
                    'std_ms': round(mp_std_ms, 2),
                    'min_ms': round(mp_min_ms, 2),
                    'max_ms': round(mp_max_ms, 2)
                } if mp_success_times else None,
                'error_types': error_types if error_types else None
            }
        }
        
        # Collect sample results
        sample_results = []
        for i, (query, result) in enumerate(zip(queries[:10], mp_results[:10])):
            sample_entry = {
                'index': i,
                'query': query,
                'status': 'success' if 'error' not in result else 'error'
            }
            if 'error' in result:
                sample_entry['error'] = result['error']
            else:
                sample_entry['actual_latitude'] = result.get('latitude')
                sample_entry['actual_longitude'] = result.get('longitude')
                sample_entry['data_points'] = len(result.get('data', []))
                if result.get('data'):
                    sample_entry['first_value'] = result['data'][0]
            sample_results.append(sample_entry)
        
        # Write output
        output_data = {
            'summary': summary,
            'sample_results': sample_results,
            'all_results_count': len(mp_results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nReport saved to: {output_file}")
        print(f"{'='*80}\n")
        
        # Add to timing report
        # add_timing_report(test_name, mp_duration, summary)
        
        # Assertions
        assert len(mp_results) == len(queries), "Should return same number of results as queries"
        assert mp_success_count > 0, f"At least some queries should succeed (got {mp_success_count})"