from typing import override, Optional, List
import json
import re
from datetime import datetime, timedelta

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from .nasa_power_data import NASAPowerDataFetcher, COMMON_PARAMETERS, enrich_locations_with_nasa_data, has_valid_nasa_power_data
from .open_street_map import geocode_address
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

DESCRIPTION = """\
This agent can do the following:
- List available NASA POWER parameters: Returns a complete list of all available NASA POWER parameters with their full descriptions. Use this to show/list/display/get all weather and climate parameters (like T2M for temperature, RH2M for humidity, PRECTOTCORR for precipitation, etc.) from the NASA POWER dataset.
- Enrich locations: Enriches or adds NASA POWER data (e.g. temperature at 2m T2M) to location records. The enriched data is written to the artifact only; each record gets nasaPowerProperties.

To use this agent, provide an artifact local_id with location records containing latitude, longitude, and date information. The agent will extract location data from the artifact and enrich it with NASA POWER weather and climate data.
"""

class BatchEnrichParams(BaseModel):

    locations_artifact: Optional[Artifact] = Field(default=None)
    address: Optional[str] = Field(default=None)
    weather_parameters: List[str] = Field(default=['T2M'])
    date_range_days: int = Field(default=1)
    frequency: str = Field(default="daily")
    source: str = Field(default="merra2")
    temporal: str = Field(default="temporal")
    time: str = Field(default="utc")


class NASAPowerAgent(IChatBioAgent):

    def __init__(self):
        super().__init__()
        self.data_fetcher = NASAPowerDataFetcher()
        self._cached_parameters = None

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="NASA POWER Data Agent",
            description="Access NASA's Prediction Of Worldwide Energy Resources (POWER) weather and climate data for any location worldwide.",
            icon="https://earthdata.nasa.gov/s3fs-public/2022-11/power_logo_event.png",
            entrypoints=[
                AgentEntrypoint(
                    id="enrich_locations",
                    description=DESCRIPTION,
                    parameters=BatchEnrichParams
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        if entrypoint == "enrich_locations":
            await self._handle_list_parameters(context, request)
            if params is not None:
                await self._handle_enrich_locations(context, request, params)
        else:
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def _handle_list_parameters(self, context: ResponseContext, request: str):
        """Handle requests to list available parameters"""
        if self._cached_parameters is not None:
            await context.reply(self._cached_parameters)
            return
        
        async with context.begin_process(summary="Enriching records with NASA POWER data (listing available parameters & enriching location records)") as process:
            process: IChatBioAgentProcess
            
            await process.log("Retrieving available parameters from NASA POWER API metadata endpoint")
            
            # Get parameter metadata from API
            param_metadata = self.data_fetcher.get_parameter_metadata()
            
            if not param_metadata:
                # Fallback to simple parameter info if API fails
                param_info = self.data_fetcher.get_parameter_info()
                param_ids = sorted(param_info.keys())
                param_ids_str = ','.join(param_ids)
                await process.log(f"Retrieved {len(param_ids)} parameters: {param_ids_str}")
                
                response = "**Available NASA POWER Parameters:**\n\n"
                for param, description in sorted(param_info.items()):
                    response += f"**{param}**\n  {description}\n\n"
            else:
                # Format parameters: extract name, keywords, and sources (all parameters, regardless of source)
                formatted_params = {}
                param_ids = []
                all_sources = set()
                for param in param_metadata:
                    param_id = param.get('id')
                    sources = param.get('sources', [])
                    
                    if param_id:
                        formatted_params[param_id] = {
                            "name": param.get('name', 'Unknown parameter'),
                            "keywords": param.get('keywords', []),
                            "sources": sources
                        }
                        param_ids.append(param_id)
                        all_sources.update(sources)
                
                # param_ids_str = ','.join(sorted(param_ids))
                sources_str = ', '.join(sorted(all_sources))
                # await process.log(f"Formatted {len(formatted_params)} parameters from all sources: {param_ids_str}")
                await process.log(f"Available sources: {sources_str}")
                
                # Categorize parameters by source
                params_by_source = {}
                for param_id, param_data in formatted_params.items():
                    sources = param_data.get('sources', [])
                    for source in sources:
                        if source not in params_by_source:
                            params_by_source[source] = []
                        params_by_source[source].append(param_id)
                
                # Log parameters grouped by source
                for source in sorted(params_by_source.keys()):
                    source_params = sorted(params_by_source[source])
                    params_list = ', '.join(source_params)
                    await process.log(f"{source} parameters ({len(source_params)}): {params_list}")
                
                # Format the response for user
                response = "**Available NASA POWER Parameters:**\n\n"
                response += f"**Available Sources:** {', '.join(sorted(all_sources))}\n\n"
                
                # Add parameters grouped by source to response
                response += "**Parameters by Source:**\n\n"
                for source in sorted(params_by_source.keys()):
                    source_params = sorted(params_by_source[source])
                    params_list = ', '.join(source_params)
                    response += f"**{source}** ({len(source_params)} parameters): {params_list}\n\n"
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
            
            response += "\nYou can use any of these parameters when enriching location records with the `enrich_locations` entrypoint."
            
            self._cached_parameters = response
            
            await context.reply(response)
    
    async def _handle_common_query(self, context: ResponseContext, request: str, params: BatchEnrichParams):
        """Handle weather data query requests without artifact (direct lat/lon/date queries)"""
        async with context.begin_process(summary="Fetching NASA POWER weather data") as process:
            process: IChatBioAgentProcess
            
            await process.log("Fetching NASA POWER weather data")
            await process.log(
                f"Data source: {params.source} (merra2 is the default data source.)"
            )
            
            # Try to extract lat/lon/date from request string, or use defaults
            # Default: New York City, today
            default_lat = 40.7128
            default_lon = -74.0060
            default_date = datetime.now().strftime("%Y-%m-%d")
            
            # Simple extraction from request (can be improved with NLP)
            latitude = default_lat
            longitude = default_lon
            start_date = default_date
            end_date = default_date
            
            # Determine address to use: from params or try to extract from request
            address_to_use = params.address
            
            # If no address in params, try to extract location name from request
            # Look for patterns like "in [Location]", "at [Location]", "for [Location]"
            if not address_to_use:
                location_patterns = [
                    r'in\s+([A-Z][a-zA-Z\s,]+?)(?:\s+at\s+|\s+on\s+|$)',
                    r'at\s+([A-Z][a-zA-Z\s,]+?)(?:\s+at\s+|\s+on\s+|$)',
                    r'for\s+([A-Z][a-zA-Z\s,]+?)(?:\s+at\s+|\s+on\s+|$)',
                ]
                for pattern in location_patterns:
                    match = re.search(pattern, request, re.IGNORECASE)
                    if match:
                        potential_address = match.group(1).strip()
                        # Remove date-like patterns from the address
                        potential_address = re.sub(r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}', '', potential_address, flags=re.IGNORECASE).strip()
                        if potential_address and len(potential_address) > 2:
                            address_to_use = potential_address
                            break
            
            # If address is available (from params or extracted), geocode it to get lat/lon
            if address_to_use:
                await process.log(f"Geocoding address: {address_to_use}")
                geocoded = geocode_address(address_to_use)
                if geocoded:
                    latitude, longitude = geocoded
                    await process.log(f"Geocoded to coordinates: ({latitude}, {longitude})")
                else:
                    await process.log(f"Warning: Failed to geocode address '{address_to_use}', falling back to default location")
            
            # Try to extract coordinates from request (only if address wasn't provided or failed)
            if not address_to_use or latitude == default_lat:
                # Look for lat/lon patterns like "40.7128, -74.0060" or "lat: 40.7128, lon: -74.0060"
                coord_pattern = r'(-?\d+\.?\d*)\s*[,:]\s*(-?\d+\.?\d*)'
                coord_match = re.search(coord_pattern, request)
                if coord_match:
                    try:
                        latitude = float(coord_match.group(1))
                        longitude = float(coord_match.group(2))
                    except ValueError:
                        pass
            
            # Try to extract date from request - handle multiple formats
            date_extracted = False
            
            # Format 1: YYYY-MM-DD
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            date_matches = re.findall(date_pattern, request)
            if date_matches:
                start_date = date_matches[0]
                end_date = date_matches[-1] if len(date_matches) > 1 else date_matches[0]
                date_extracted = True
                await process.log(f"Extracted date (YYYY-MM-DD format): {start_date}")
            else:
                # Format 2: DD MMM YYYY or DD Month YYYY (e.g., "25 Dec 2024", "25 December 2024")
                date_pattern2 = r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})'
                date_match2 = re.search(date_pattern2, request, re.IGNORECASE)
                if date_match2:
                    try:
                        day = int(date_match2.group(1))
                        month_name = date_match2.group(2).lower()
                        year = int(date_match2.group(3))
                        
                        month_map = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        month = month_map.get(month_name[:3], 1)
                        
                        date_obj = datetime(year, month, day)
                        start_date = date_obj.strftime("%Y-%m-%d")
                        end_date = start_date
                        date_extracted = True
                        await process.log(f"Extracted date (DD MMM YYYY format): {start_date} (from '{date_match2.group(0)}')")
                    except ValueError as e:
                        await process.log(f"Failed to parse extracted date: {str(e)}")
                else:
                    # Format 3: Month DD, YYYY (e.g., "December 25, 2024", "Dec 25, 2024")
                    date_pattern3 = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})'
                    date_match3 = re.search(date_pattern3, request, re.IGNORECASE)
                    if date_match3:
                        try:
                            month_name = date_match3.group(1).lower()
                            day = int(date_match3.group(2))
                            year = int(date_match3.group(3))
                            
                            month_map = {
                                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                            }
                            month = month_map.get(month_name[:3], 1)
                            
                            date_obj = datetime(year, month, day)
                            start_date = date_obj.strftime("%Y-%m-%d")
                            end_date = start_date
                            date_extracted = True
                            await process.log(f"Extracted date (Month DD, YYYY format): {start_date} (from '{date_match3.group(0)}')")
                        except ValueError as e:
                            await process.log(f"Failed to parse extracted date: {str(e)}")
                    else:
                        await process.log(f"Could not find date pattern in request: '{request}'")
            
            if not date_extracted:
                await process.log(f"No date found in request, using default date: {default_date}")
            
            # Use first parameter from weather_parameters list
            parameter = params.weather_parameters[0] if params.weather_parameters else 'T2M'
            
            await process.log(
                f"Querying NASA POWER data:\n"
                f"  Location: ({latitude}, {longitude})\n"
                f"  Parameter: {parameter}\n"
                f"  Date Range: {start_date} to {end_date}\n"
                f"  Frequency: {params.frequency}"
            )
            
            try:
                # Query NASA POWER data
                data = self.data_fetcher.get_data_from_zarr_with_xarray(
                    start_date=start_date,
                    end_date=end_date,
                    latitude=latitude,
                    longitude=longitude,
                    parameter=parameter,
                    frequency=params.frequency,
                    source=params.source,
                    temporal=params.temporal,
                    time=params.time
                )
                
                num_data_points = len(data['data'])
                await process.log(f"Successfully retrieved {num_data_points} data points")
                
                # Build summary response
                summary = (
                    f"**NASA POWER Data Summary**\n\n"
                    f"**Location:** ({data['latitude']}, {data['longitude']})\n"
                    f"**Parameter:** {data['parameter']} - {data['parameter_description']}\n"
                    f"**Frequency:** {data['frequency']}\n"
                    f"**Data Points:** {num_data_points}\n\n"
                )
                
                # Show sample data points
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
                else:
                    summary += "No data points available for the specified date range.\n"
                
                await context.reply(summary)
                
            except ValueError as e:
                await process.log(f"Validation error: {str(e)}")
                await context.reply(f"Error: {str(e)}")
            except KeyError as e:
                await process.log(f"Parameter not found: {str(e)}")
                await context.reply(f"Error: Parameter '{parameter}' not found. {str(e)}")
            except Exception as e:
                await process.log(f"Unexpected error: {str(e)}")
                await context.reply(f"An error occurred while fetching data: {str(e)}")
    
    async def _handle_enrich_locations(self, context: ResponseContext, request: str, params: Optional[BatchEnrichParams]):
        """Handle batch enrichment of location records with NASA POWER data"""
        async with context.begin_process(summary="Enriching records with NASA POWER data") as process:
            process: IChatBioAgentProcess
            
            if params is None:
                await context.reply("Error: No parameters provided. Please provide locations data.")
                return
            
            locations = None

            if params.locations_artifact is None:
                await self._handle_common_query(context, request, params)
                return

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

            await process.log(
                f"Data source: {params.source} (merra2 is the default data source.)"
            )

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
                
                # Check for errors in enriched locations
                errors_found = []
                for i, loc in enumerate(enriched_locations):
                    nasa_props = loc.get('nasaPowerProperties')
                    if nasa_props and isinstance(nasa_props, list):
                        for prop in nasa_props:
                            if isinstance(prop, dict) and 'error' in prop:
                                error_msg = prop.get('error', 'Unknown error')
                                param = prop.get('parameter', 'Unknown parameter')
                                errors_found.append(f"Location {i+1} ({param}): {error_msg}")
                
                artifact_locations = []
                for loc in enriched_locations:
                    rec = dict(loc)
                    if not has_valid_nasa_power_data(loc):
                        rec["nasaPowerProperties"] = None
                    artifact_locations.append(rec)

                log_msg = f"Successfully enriched {successful} records\nSkipped {skipped} records (missing data)"
                if errors_found:
                    # Log first few errors as examples
                    error_samples = errors_found[:5]
                    log_msg += f"\n\nErrors encountered ({len(errors_found)} total):"
                    for err in error_samples:
                        log_msg += f"\n  - {err}"
                    if len(errors_found) > 5:
                        log_msg += f"\n  ... and {len(errors_found) - 5} more errors"
                
                await process.log(log_msg)
                await process.log(
                    f"Enriched {successful} location records with NASA POWER data (out of {len(enriched_locations)} total)",
                    data={
                        "source": [params.source] if params.source else [],
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
                    f"**Source:** {params.source}\n"
                    f"**Date Range:** {date_info}\n"
                    f"**Frequency:** {frequency}\n"
                )
                
                if errors_found:
                    summary += f"\n**Errors Encountered:** {len(errors_found)} error(s) occurred during data fetching.\n"
                    summary += "Common causes: parameter not available in specified source, invalid Zarr URL, or data unavailable for the date range.\n"
                    if len(errors_found) <= 3:
                        summary += "\n**Error Details:**\n"
                        for err in errors_found:
                            summary += f"  - {err}\n"
                    else:
                        summary += f"\n**Sample Errors (showing first 3 of {len(errors_found)}):**\n"
                        for err in errors_found[:3]:
                            summary += f"  - {err}\n"
                    summary += "\n"

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
