from typing import override, Optional, List
import json
from datetime import datetime, timedelta

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from .nasa_power_data import NASAPowerDataFetcher, COMMON_PARAMETERS, enrich_locations_with_nasa_data, has_valid_nasa_power_data
from .util import (
    retrieve_artifact_content,
    parse_locations_json,
    extract_json_schema,
    select_location_properties,
    LocationPropertyPaths,
    LocationStringPaths,
    GeoJSONCoordinatesPaths,
    GiveUp,
    read_path,
    parse_location_string,
    parse_geojson_coordinates,
    try_extract_locations_heuristic,
)

class NASAPowerQueryParams(BaseModel):
    """Parameters for querying NASA POWER data"""
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    locations: Optional[List[dict[str, float]]] = Field(None)
    parameter: str = Field(default="T2M")
    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)
    frequency: str = Field(default="daily")
    source: str = Field(default="merra2")
    temporal: str = Field(default="temporal")
    time: str = Field(default="utc")

class BatchEnrichParams(BaseModel):
    """Parameters for batch enrichment of location records"""

    locations_artifact: Optional[Artifact] = Field(default=None)
    # locations_json: Optional[str] = Field(default=None)
    weather_parameters: List[str] = Field(default=['T2M'])
    date_range_days: int = Field(default=1)
    frequency: str = Field(default="daily")
    source: str = Field(default="merra2")
    temporal: str = Field(default="temporal")
    time: str = Field(default="utc")


class NASAPowerAgent(IChatBioAgent):
    """
    NASA POWER Data Agent - Provides access to NASA's weather and climate data.
    """

    def __init__(self):
        super().__init__()
        self.data_fetcher = NASAPowerDataFetcher()

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="NASA POWER Data Agent",
            description="Access NASA's Prediction Of Worldwide Energy Resources (POWER) weather and climate data for any location worldwide.",
            icon="https://earthdata.nasa.gov/s3fs-public/2022-11/power_logo_event.png",
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="query_weather",
                    description="",
                    # description=(
                    #     "Query weather and climate data for one or more locations and time periods. "
                    #     "Use this for ALL questions about weather data (e.g., 'What was the temperature in X?', 'Show me weather for multiple cities', 'Get weather data for date ranges'). "
                    #     "Supports single location queries, date ranges (e.g., '20-25 December 2024'), and multiple locations. "
                    #     "For date ranges or multiple locations, automatically uses batch processing for optimal performance. "
                    #     "Returns a human-readable summary with sample data points. "
                    #     "This is the PRIMARY entrypoint for weather queries and questions. "
                    #     "Parameters: "
                    #     "latitude - Latitude (-90 to 90). Use this for single location queries. "
                    #     "longitude - Longitude (-180 to 180). Use this for single location queries. "
                    #     "locations - List of locations with 'latitude' and 'longitude' keys. Use this for multiple location queries. "
                    #     "parameter - Weather parameter to query (e.g., T2M, RH2M). "
                    #     "start_date - Start date in YYYY-MM-DD format. "
                    #     "end_date - End date in YYYY-MM-DD format. "
                    #     "frequency - Data frequency: hourly, daily, or monthly."
                    # ),
                    parameters=NASAPowerQueryParams
                ),
                AgentEntrypoint(
                    id="list_parameters",
                    description="Returns a complete list of all available NASA POWER parameters with their full descriptions. Use this to show/list/display/get all weather and climate parameters (like T2M for temperature, RH2M for humidity, PRECTOTCORR for precipitation, etc.) from the NASA POWER dataset. Call this when users ask 'what parameters', 'list parameters', 'show parameters', 'available parameters', or similar questions about NASA POWER data fields.",
                    parameters=None
                ),
                AgentEntrypoint(
                    id="enrich_locations",
                    description=(
                        "Enriches or adds NASA POWER data (e.g. temperature at 2m T2M/T2M_MAX/T2M_MIN, humidity RH2M, wind speed WS2M, surface pressure PS) to location records. "
                        "The enriched data is written to the artifact only; each record gets nasaPowerProperties. "
                        "Parameters: "
                        "locations_artifact - Use this when referencing an EXISTING artifact (e.g., #xxxx from a previous agent response or file upload). This is the DEFAULT and PREFERRED method for production use. "
                        # "locations_json - ONLY use this for TESTING or TEST PURPOSE when the user explicitly mentions testing, test purpose, or provides JSON directly in chat. Pass the JSON array string containing location records with eventDate, decimalLatitude, and decimalLongitude fields. "
                        "weather_parameters - List of NASA POWER parameters to fetch (e.g., T2M, RH2M). "
                        "date_range_days - Number of days to fetch (1 = exact event date only, >1 = range around event). (Default: 1)"
                        "frequency - Data frequency: hourly, daily, or monthly. (Default: daily)"
                    ),
                    parameters=BatchEnrichParams
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        if entrypoint == "query_weather":
            await self._handle_query_weather(context, request, params)
        elif entrypoint == "list_parameters":
            await self._handle_list_parameters(context, request)
        elif entrypoint == "enrich_locations":
            await self._handle_enrich_locations(context, request, params)
        else:
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def _handle_query_weather(self, context: ResponseContext, request: str, params: Optional[NASAPowerQueryParams]):
        """Handle weather data query requests"""
        async with context.begin_process(summary="Fetching NASA POWER weather data") as process:
            process: IChatBioAgentProcess
            
            # Set default parameters if not provided
            if params is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                params = NASAPowerQueryParams(
                    latitude=40.7128,
                    longitude=-74.0060,
                    parameter="T2M",
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    frequency="daily"
                )
                await process.log(f"Using default parameters: New York City, last 30 days, T2M (temperature)")
            
            # Set default dates if not provided
            if not params.start_date or not params.end_date:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                params.start_date = start_date.strftime("%Y-%m-%d")
                params.end_date = end_date.strftime("%Y-%m-%d")
            
            # Parse dates
            try:
                start_dt = datetime.strptime(params.start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(params.end_date, "%Y-%m-%d")
            except ValueError:
                await context.reply("Error: Dates must be in YYYY-MM-DD format")
                return
            
            # Determine if we have multiple locations
            locations = []
            if params.locations:
                locations = params.locations
            elif params.latitude is not None and params.longitude is not None:
                locations = [{"latitude": params.latitude, "longitude": params.longitude}]
            else:
                await context.reply("Error: Must provide either latitude/longitude or locations list")
                return
            
            # Determine if we need batch processing
            # Use batch if: date range (multiple days) OR multiple locations
            use_batch = False
            if len(locations) > 1:
                use_batch = True
                await process.log(f"Multiple locations detected ({len(locations)}), using batch processing")
            elif start_dt != end_dt and params.frequency == "daily":
                # For date ranges with daily frequency, split into individual day queries
                days_diff = (end_dt - start_dt).days + 1
                if days_diff > 1:
                    use_batch = True
                    await process.log(f"Date range detected ({days_diff} days), using batch processing")
            
            try:
                if use_batch:
                    # Prepare batch queries
                    queries = []
                    
                    if len(locations) > 1:
                        # Multiple locations: create query for each location with date range
                        for loc in locations:
                            queries.append({
                                'start_date': params.start_date,
                                'end_date': params.end_date,
                                'latitude': loc['latitude'],
                                'longitude': loc['longitude'],
                                'parameter': params.parameter,
                                'frequency': params.frequency,
                                'source': params.source,
                                'temporal': params.temporal,
                                'time': params.time
                            })
                    else:
                        # Single location but date range: create query for each day
                        current_date = start_dt
                        while current_date <= end_dt:
                            date_str = current_date.strftime("%Y-%m-%d")
                            queries.append({
                                'start_date': date_str,
                                'end_date': date_str,
                                'latitude': locations[0]['latitude'],
                                'longitude': locations[0]['longitude'],
                                'parameter': params.parameter,
                                'frequency': params.frequency,
                                'source': params.source,
                                'temporal': params.temporal,
                                'time': params.time
                            })
                            current_date += timedelta(days=1)
                    
                    await process.log(
                        f"Processing {len(queries)} queries using batch multiprocessing:\n\n"
                        f"  Locations: {len(locations)}\n"
                        f"  Parameter: {params.parameter}\n"
                        f"  Date Range: {params.start_date} to {params.end_date}\n"
                        f"  Frequency: {params.frequency}"
                    )
                    
                    # Execute batch queries
                    results = self.data_fetcher.get_data_from_zarr_batch_multiprocessing(queries)
                    
                    # Process results
                    successful_results = [r for r in results if 'error' not in r]
                    failed_results = [r for r in results if 'error' in r]
                    
                    await process.log(
                        f"Batch processing complete:\n"
                        f"  Successful: {len(successful_results)}\n"
                        f"  Failed: {len(failed_results)}"
                    )
                    
                    # Combine results for single location date range
                    if len(locations) == 1 and len(queries) > 1:
                        # Merge all data points from different days
                        all_data_points = []
                        for result in successful_results:
                            if 'data' in result:
                                all_data_points.extend(result['data'])
                        
                        # Sort by date
                        all_data_points.sort(key=lambda x: x['date'])
                        
                        # Create combined result
                        if successful_results:
                            # Get parameter description from API
                            param_info = self.data_fetcher.get_parameter_info()
                            param_description = param_info.get(params.parameter, successful_results[0].get('parameter_description', 'Unknown parameter'))
                            
                            combined_data = {
                                'parameter': params.parameter,
                                'parameter_description': param_description,
                                'frequency': params.frequency,
                                'latitude': successful_results[0]['latitude'],
                                'longitude': successful_results[0]['longitude'],
                                'data': all_data_points,
                                'source': 'zarr_xarray_batch'
                            }
                            
                            await process.create_artifact(
                                mimetype="application/json",
                                description=f"NASA POWER {params.parameter} data for ({locations[0]['latitude']}, {locations[0]['longitude']})",
                                content=json.dumps(combined_data, indent=2).encode('utf-8'),
                                metadata={
                                    "source": "NASA POWER",
                                    "parameter": params.parameter,
                                    "location": f"{locations[0]['latitude']},{locations[0]['longitude']}",
                                    "frequency": params.frequency,
                                    "query_count": len(queries)
                                }
                            )
                            
                            num_data_points = len(all_data_points)
                            summary = (
                                f"**NASA POWER Data Summary (Batch Processing)**\n\n"
                                f"**Location:** ({combined_data['latitude']}, {combined_data['longitude']})\n"
                                f"**Parameter:** {combined_data['parameter']} - {combined_data['parameter_description']}\n"
                                f"**Frequency:** {combined_data['frequency']}\n"
                                f"**Data Points:** {num_data_points}\n"
                                f"**Queries Processed:** {len(queries)}\n\n"
                            )
                            
                            if num_data_points > 0:
                                summary += "**Sample Data:**\n"
                                sample_count = min(5, num_data_points)
                                for item in all_data_points[:sample_count]:
                                    value_str = f"{item['value']:.2f}" if item['value'] is not None else "null"
                                    summary += f"  - {item['date']}: {value_str}\n"
                                if num_data_points > sample_count * 2:
                                    summary += "  ...\n"
                                    for item in all_data_points[-sample_count:]:
                                        value_str = f"{item['value']:.2f}" if item['value'] is not None else "null"
                                        summary += f"  - {item['date']}: {value_str}\n"
                            
                            if failed_results:
                                summary += f"\n{len(failed_results)} queries failed."
                            
                            await context.reply(summary)
                        else:
                            await context.reply(f"Error: All queries failed. First error: {failed_results[0]['error'] if failed_results else 'Unknown error'}")
                    else:
                        # Multiple locations: create summary for each
                        all_results = []
                        for i, result in enumerate(successful_results):
                            if 'error' not in result:
                                all_results.append(result)
                        
                        await process.create_artifact(
                            mimetype="application/json",
                            description=f"NASA POWER {params.parameter} data for {len(locations)} locations",
                            content=json.dumps(all_results, indent=2).encode('utf-8'),
                            metadata={
                                "source": "NASA POWER",
                                "parameter": params.parameter,
                                "location_count": len(locations),
                                "frequency": params.frequency,
                                "query_count": len(queries)
                            }
                        )
                        
                        summary = (
                            f"**NASA POWER Data Summary (Batch Processing)**\n\n"
                            f"**Locations:** {len(locations)}\n"
                            f"**Parameter:** {params.parameter}\n"
                            f"**Date Range:** {params.start_date} to {params.end_date}\n"
                            f"**Frequency:** {params.frequency}\n"
                            f"**Successful Queries:** {len(successful_results)}\n"
                            f"**Failed Queries:** {len(failed_results)}\n\n"
                        )
                        
                        # Show sample results for first few locations
                        if successful_results:
                            summary += "**Sample Results:**\n"
                            for i, result in enumerate(successful_results[:3]):
                                loc = locations[i] if i < len(locations) else {}
                                summary += f"\n**Location {i+1}:** ({result.get('latitude', loc.get('latitude', 'N/A'))}, {result.get('longitude', loc.get('longitude', 'N/A'))})\n"
                                if 'data' in result and len(result['data']) > 0:
                                    sample_data = result['data'][0]
                                    value_str = f"{sample_data['value']:.2f}" if sample_data['value'] is not None else "null"
                                    summary += f"  - {sample_data['date']}: {value_str}\n"
                                summary += f"  - Total data points: {len(result.get('data', []))}\n"
                            
                            if len(successful_results) > 3:
                                summary += f"\n... and {len(successful_results) - 3} more locations\n"
                        
                        summary += "\nFull results available in the JSON artifact."
                        
                        await context.reply(summary)
                else:
                    # Single query (single location, single date or already handled date range)
                    await process.log(
                        f"Querying NASA POWER data:\n\n"
                        f"  Location: ({locations[0]['latitude']}, {locations[0]['longitude']})\n\n"
                        f"  Parameter: {params.parameter}\n\n"
                        f"  Date Range: {params.start_date} to {params.end_date}\n\n"
                        f"  Frequency: {params.frequency}"
                    )
                    
                    data = self.data_fetcher.get_data_from_zarr_with_xarray(
                        start_date=params.start_date,
                        end_date=params.end_date,
                        latitude=locations[0]['latitude'],
                        longitude=locations[0]['longitude'],
                        parameter=params.parameter,
                        frequency=params.frequency,
                        source=params.source,
                        temporal=params.temporal,
                        time=params.time
                    )
                    
                    num_data_points = len(data['data'])
                    await process.log(f"Successfully retrieved {num_data_points} data points")
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"NASA POWER {params.parameter} data for ({locations[0]['latitude']}, {locations[0]['longitude']})",
                        content=json.dumps(data, indent=2).encode('utf-8'),
                        metadata={
                            "source": "NASA POWER",
                            "parameter": params.parameter,
                            "location": f"{locations[0]['latitude']},{locations[0]['longitude']}",
                            "frequency": params.frequency
                        }
                    )
                    
                    summary = (
                        f"**NASA POWER Data Summary**\n\n"
                        f"**Location:** ({data['latitude']}, {data['longitude']})\n"
                        f"**Parameter:** {data['parameter']} - {data['parameter_description']}\n"
                        f"**Frequency:** {data['frequency']}\n"
                        f"**Data Points:** {num_data_points}\n\n"
                    )
                    
                    # Show first few and last few data points
                    if num_data_points > 0:
                        summary += "**Sample Data:**\n"
                        sample_count = min(5, num_data_points)
                        for item in data['data'][:sample_count]:
                            value_str = f"{item['value']:.2f}" if item['value'] is not None else "null"
                            summary += f"  - {item['date']}: {value_str}\n"
                        if num_data_points > sample_count * 2:
                            summary += "  ...\n"
                            for item in data['data'][-sample_count:]:
                                value_str = f"{item['value']:.2f}" if item['value'] is not None else "null"
                                summary += f"  - {item['date']}: {value_str}\n"
                    
                    await context.reply(summary)
                
            except ValueError as e:
                await process.log(f"Validation error: {str(e)}")
                await context.reply(f"Error: {str(e)}")
            except KeyError as e:
                await process.log(f"Parameter not found: {str(e)}")
                await context.reply(f"Error: {str(e)}")
            except Exception as e:
                await process.log(f"Unexpected error: {str(e)}")
                await context.reply(f"An error occurred while fetching data: {str(e)}")

    async def _handle_list_parameters(self, context: ResponseContext, request: str):
        """Handle requests to list available parameters"""
        async with context.begin_process(summary="Listing available NASA POWER parameters") as process:
            process: IChatBioAgentProcess
            
            await process.log("Retrieving available parameters from NASA POWER API metadata endpoint")
            
            # Get parameter metadata from API
            param_metadata = self.data_fetcher.get_parameter_metadata()
            
            if not param_metadata:
                # Fallback to simple parameter info if API fails
                param_info = self.data_fetcher.get_parameter_info()
                response = "**Available NASA POWER Parameters:**\n\n"
                for param, description in sorted(param_info.items()):
                    response += f"**{param}**\n  {description}\n\n"
            else:
                # Format parameters: extract only name, keywords, and sources
                formatted_params = {}
                for param in param_metadata:
                    param_id = param.get('id')
                    sources = param.get('sources', [])
                    
                    if param_id and 'merra2' in sources:
                        formatted_params[param_id] = {
                            "name": param.get('name', 'Unknown parameter'),
                            "keywords": param.get('keywords', [])
                        }
                
                # Process log the formatted parameters
                await process.log(
                    f"Formatted {len(formatted_params)} parameters",
                    data={"formatted_parameters": formatted_params}
                )
                
                # Format the response for user
                response = "**Available NASA POWER Parameters:**\n\n"
                for param_id, param_data in sorted(formatted_params.items()):
                    param_name = param_data.get('name', 'Unknown parameter')
                    keywords = param_data.get('keywords', [])
                    sources = param_data.get('sources', [])
                    
                    response += f"**{param_id}** - {param_name}\n"
                    if keywords:
                        response += f"  Keywords: {', '.join(keywords)}\n"
                    if sources:
                        response += f"  Sources: {', '.join(sources)}\n"
                    response += "\n"
            
            response += "\nYou can query any of these parameters using the `query_weather` entrypoint."
            
            await context.reply(response)
    
    async def _handle_enrich_locations(self, context: ResponseContext, request: str, params: Optional[BatchEnrichParams]):
        """Handle batch enrichment of location records with NASA POWER data"""
        async with context.begin_process(summary="Enriching location records with NASA POWER data") as process:
            process: IChatBioAgentProcess
            
            if params is None:
                await context.reply("Error: No parameters provided. Please provide locations data.")
                return
            
            locations = None
            
            # Only use locations_json for testing purposes
            # if params.locations_json is not None:
            #     await process.log("Using locations from provided JSON string")
                
            #     # Parse the JSON content to get locations array
            #     try:
            #         locations = await parse_locations_json(params.locations_json, process)
            #     except json.JSONDecodeError as e:
            #         await context.reply(f"Error: Invalid JSON in location data: {str(e)}. Please ensure the JSON is a valid array of location records.")
            #         return
            #     except ValueError as e:
            #         await context.reply(f"Error: {str(e)}")
            #         return
            if params.locations_artifact is not None:
                try:
                    locations = await retrieve_artifact_content(params.locations_artifact, process)
                    
                    # Extract schema and select location properties
                    schema = extract_json_schema(locations)
                    await process.log("Extracted JSON schema from artifact content")
                    
                    selection_result = await select_location_properties(request, schema)
                    
                    match selection_result:
                        case LocationPropertyPaths() as paths:
                            await process.log(
                                "Using scalar field paths for location extraction",
                                data={
                                    "format": "scalar_fields",
                                    "latitude": paths.latitude,
                                    "longitude": paths.longitude,
                                    "date": paths.date,
                                },
                            )
                            
                            # Extract values using the paths
                            latitudes = list(read_path(locations, paths.latitude))
                            longitudes = list(read_path(locations, paths.longitude))
                            dates = list(read_path(locations, paths.date))
                            
                            # Reconstruct locations array with standard field names
                            locations = []
                            for lat, lon, date in zip(latitudes, longitudes, dates):
                                if lat is not None and lon is not None and date is not None:
                                    locations.append({
                                        "decimalLatitude": float(lat) if lat is not None else None,
                                        "decimalLongitude": float(lon) if lon is not None else None,
                                        "eventDate": str(date) if date is not None else None,
                                    })
                            
                            await process.log(f"Extracted {len(locations)} location records using scalar field paths")
                        
                        case LocationStringPaths() as paths:
                            await process.log(
                                "Using location string path for location extraction",
                                data={
                                    "format": "location_string",
                                    "location_string": paths.location_string,
                                    "date": paths.date,
                                },
                            )
                            
                            # Extract values using the paths
                            location_strings = list(read_path(locations, paths.location_string))
                            dates = list(read_path(locations, paths.date))
                            
                            # Parse location strings and reconstruct locations array
                            locations = []
                            for loc_str, date in zip(location_strings, dates):
                                parsed = parse_location_string(loc_str)
                                if parsed is not None and date is not None:
                                    lat, lon = parsed
                                    locations.append({
                                        "decimalLatitude": lat,
                                        "decimalLongitude": lon,
                                        "eventDate": str(date) if date is not None else None,
                                    })
                            
                            await process.log(f"Extracted {len(locations)} location records using location string format")
                        
                        case GeoJSONCoordinatesPaths() as paths:
                            await process.log(
                                "Using GeoJSON coordinates path for location extraction",
                                data={
                                    "format": "geojson_coordinates",
                                    "coordinates": paths.coordinates,
                                    "date": paths.date,
                                },
                            )
                            
                            # Extract values using the paths
                            coordinates_list = list(read_path(locations, paths.coordinates))
                            dates = list(read_path(locations, paths.date))
                            
                            # Parse GeoJSON coordinates and reconstruct locations array
                            locations = []
                            for coords, date in zip(coordinates_list, dates):
                                parsed = parse_geojson_coordinates(coords)
                                if parsed is not None and date is not None:
                                    lat, lon = parsed
                                    locations.append({
                                        "decimalLatitude": lat,
                                        "decimalLongitude": lon,
                                        "eventDate": str(date) if date is not None else None,
                                    })
                            
                            await process.log(f"Extracted {len(locations)} location records using GeoJSON coordinates format")
                            
                        case GiveUp(reason=reason):
                            await process.log(f"Failed to identify location property paths: {reason}")
                            await process.log(
                                "Attempting heuristic extraction for iNaturalist-like JSON "
                                "(geojson.coordinates, location string, observed_on/time_observed_at)"
                            )
                            extracted = try_extract_locations_heuristic(locations)
                            if extracted is not None:
                                locations = extracted
                                await process.log(
                                    f"Extracted {len(locations)} location records using heuristic fallback"
                                )
                            else:
                                await context.reply(
                                    "Error: Could not extract location records from the artifact. "
                                    "The schema supports: separate latitude/longitude fields, a 'latitude,longitude' "
                                    "string (e.g. `location`), or GeoJSON `coordinates` [lon,lat]. "
                                    "A heuristic for iNaturalist-like JSON (e.g. `results[].geojson.coordinates`, "
                                    "`results[].location`, `observed_on`) also did not find any valid records."
                                )
                                return
                            
                except ValueError:
                    await context.reply("Error: Failed to retrieve the locations artifact content.")
                    return
            else:
                await context.reply("Error: No location data provided. Please provide either locations_artifact.")
                return
            
            # Validate that locations is a list
            if not isinstance(locations, list):
                await context.reply("Error: Location data must be a JSON array of location records")
                return
            
            if not locations:
                await context.reply("Error: Locations array is empty")
                return
            
            weather_parameters = params.weather_parameters
            date_range_days = params.date_range_days
            frequency = params.frequency
            
            if date_range_days == 1:
                date_info = "exact event date only"
            else:
                date_info = f"±{date_range_days//2} days around event"
            
            await process.log(
                f"Processing {len(locations)} location records\n"
                f"Parameters: {', '.join(weather_parameters)}\n"
                f"Date range: {date_info}\n"
                f"Frequency: {frequency}"
            )
            
            try:
                # Enrich the locations
                enriched_locations = enrich_locations_with_nasa_data(
                    locations=locations,
                    parameters=weather_parameters,
                    date_range_days=date_range_days,
                    frequency=frequency,
                    source=params.source,
                    temporal=params.temporal,
                    time=params.time
                )
                
                successful = sum(1 for loc in enriched_locations if has_valid_nasa_power_data(loc))
                skipped = len(enriched_locations) - successful
                valid_enriched_locations = [loc for loc in enriched_locations if has_valid_nasa_power_data(loc)]
                artifact_locations = []
                for loc in enriched_locations:
                    rec = dict(loc)
                    if not has_valid_nasa_power_data(loc):
                        rec["nasaPowerProperties"] = None
                    artifact_locations.append(rec)

                await process.log(
                    f"Successfully enriched {successful} records\n"
                    f"Skipped {skipped} records (missing data)"
                )
                await process.log(
                    f"Enriched {successful} location records with NASA POWER data (out of {len(enriched_locations)} total)",
                    data={
                        "source": "NASA POWER",
                        "parameters": weather_parameters,
                        "frequency": frequency,
                        "total_records": len(enriched_locations),
                        "enriched_records": successful
                    }
                )
                # Log each location's data
                for i, loc in enumerate(enriched_locations):
                    event_date = loc.get('eventDate', 'N/A')
                    lat = loc.get('decimalLatitude')
                    lon = loc.get('decimalLongitude')
                    nasa_props = loc.get('nasaPowerProperties')
                    
                    location_info = f"Location {i+1}: Date={event_date}, Lat={lat}, Lon={lon}"
                    
                    if nasa_props and isinstance(nasa_props, list) and len(nasa_props) > 0:
                        param_values = []
                        for prop in nasa_props:
                            if isinstance(prop, dict):
                                param = prop.get('parameter', 'N/A')
                                param_desc = prop.get('parameter_description', '')
                                data_list = prop.get('data', [])
                                if data_list and len(data_list) > 0:
                                    value = data_list[0].get('value')
                                    date = data_list[0].get('date', 'N/A')
                                    if value is not None:
                                        param_values.append(f"{param}={value:.2f} ({date})")
                                    else:
                                        param_values.append(f"{param}=N/A ({date})")
                        
                        if param_values:
                            location_info += f"\n  Parameters: {', '.join(param_values)}"
                        else:
                            location_info += "\n  No parameter data available"
                    else:
                        location_info += "\n  No NASA POWER data available"
                    
                    # await process.log(location_info)
                
                # Create artifact from enriched locations data (all records; nasaPowerProperties = null where data has null values)
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Location records enriched with NASA POWER data",
                        content=json.dumps(artifact_locations).encode("utf-8"),
                        metadata={
                            "format": "json",
                            "source": "NASA POWER",
                            "parameters": weather_parameters,
                            "frequency": frequency,
                            "total_records": len(enriched_locations),
                            "enriched_records": successful
                        }
                    )
                    await process.log(f"Created artifact with {len(artifact_locations)} location records ({successful} enriched, {skipped} skipped)")
                except Exception as e:
                    await process.log(f"Warning: Failed to create artifact: {str(e)}")
                
                # Format a summary
                if date_range_days == 1:
                    date_info = "exact event date only"
                else:
                    date_info = f"±{date_range_days//2} days around event"
                
                summary = (
                    f"**Batch Enrichment Complete**\n\n"
                    f"**Total Records:** {len(enriched_locations)}\n"
                    f"**Successfully Enriched:** {successful}\n"
                    f"**Skipped (missing data):** {skipped}\n"
                    f"**Parameters:** {', '.join(weather_parameters)}\n"
                    f"**Date Range:** {date_info}\n"
                    f"**Frequency:** {frequency}\n\n"
                )

                # Show sample enriched records (only those with valid non-null data)
                valid_records = valid_enriched_locations
                if valid_records:
                    summary += "**Sample Enriched Record:**\n"
                    sample = valid_records[0]
                    summary += f"  - Event Date: {sample.get('eventDate')}\n"
                    summary += f"  - Location: ({sample.get('decimalLatitude')}, {sample.get('decimalLongitude')})\n"
                    if sample.get('nasaPowerProperties'):
                        nasa_data = sample['nasaPowerProperties'][0]
                        if 'data' in nasa_data and len(nasa_data['data']) > 0:
                            first_data = nasa_data['data'][0]
                            value_str = f"{first_data['value']:.2f}" if first_data['value'] is not None else "null"
                            summary += f"  - {nasa_data.get('parameter')}: {first_data['date']} = {value_str}\n"
                
                summary += "\nArtifact with enriched location records is available."
                
                await context.reply(summary)
                
            except Exception as e:
                await process.log(f"Error during batch enrichment: {str(e)}")
                await context.reply(f"Error during batch enrichment: {str(e)}")


def create_app() -> Starlette:
    agent = NASAPowerAgent()
    app = build_agent_app(agent)
    return app
