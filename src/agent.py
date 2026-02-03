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

DESCRIPTION = """\
This agent can do the following:
- List available NASA POWER parameters: Returns a complete list of all available NASA POWER parameters with their full descriptions. Use this to show/list/display/get all weather and climate parameters (like T2M for temperature, RH2M for humidity, PRECTOTCORR for precipitation, etc.) from the NASA POWER dataset.
- Enrich locations: Enriches or adds NASA POWER data (e.g. temperature at 2m T2M) to location records. The enriched data is written to the artifact only; each record gets nasaPowerProperties.

To use this agent, provide an artifact local_id with location records containing latitude, longitude, and date information. The agent will extract location data from the artifact and enrich it with NASA POWER weather and climate data.
"""

class BatchEnrichParams(BaseModel):

    locations_artifact: Optional[Artifact] = Field(default=None)
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
        
        async with context.begin_process(summary="Listing available NASA POWER parameters") as process:
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
                # Format parameters: extract only name, keywords, and sources
                formatted_params = {}
                param_ids = []
                for param in param_metadata:
                    param_id = param.get('id')
                    sources = param.get('sources', [])
                    
                    if param_id and 'merra2' in sources:
                        formatted_params[param_id] = {
                            "name": param.get('name', 'Unknown parameter'),
                            "keywords": param.get('keywords', [])
                        }
                        param_ids.append(param_id)
                
                param_ids_str = ','.join(sorted(param_ids))
                await process.log(f"Formatted {len(formatted_params)} parameters: {param_ids_str}")
                
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
            
            response += "\nYou can use any of these parameters when enriching location records with the `enrich_locations` entrypoint."
            
            self._cached_parameters = response
            
            await context.reply(response)
    
    async def _handle_enrich_locations(self, context: ResponseContext, request: str, params: Optional[BatchEnrichParams]):
        """Handle batch enrichment of location records with NASA POWER data"""
        async with context.begin_process(summary="Enriching location records with NASA POWER data") as process:
            process: IChatBioAgentProcess
            
            if params is None:
                await context.reply("Error: No parameters provided. Please provide locations data.")
                return
            
            locations = None
            
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
