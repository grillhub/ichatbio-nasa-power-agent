from typing import override, Optional, List
import json
import re
from datetime import UTC, datetime, timedelta

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from nasa_power_data import (
    NASAPowerDataFetcher,
    _count_valid_nasa_date_rows,
    enrich_locations_with_nasa_data,
)
from open_street_map import geocode_address
from utils.procedures import (
    extract_json_schema,
    retrieve_artifact_content,
    extract_map_data_from_json,
)
from utils.dateUtil import (
    sanitize_locations
)

def _count_artifact_records(source_content: dict | list) -> int:
    if isinstance(source_content, list):
        return len(source_content)
    if isinstance(source_content, dict):
        items = source_content.get("items")
        results = source_content.get("results")
        if isinstance(items, list):
            return len(items)
        if isinstance(results, list):
            return len(results)
    return 0


DESCRIPTION = """\
This agent can do the following:
 - List available NASA POWER parameters: Returns a complete list of all available NASA POWER parameters with their full descriptions. Use this to show/list/display/get all weather and climate parameters (like T2M for temperature, RH2M for humidity, PRECTOTCORR for precipitation, etc.) from the NASA POWER dataset.
 - Enrich locations: Enriches or adds NASA POWER data (e.g. temperature at 2m T2M) to location records. The enriched data is written to the artifact only; each record gets nasaPowerProperties.
 - Query common question: Converts a natural language query or address string into a JSON list of locations (including latitude, longitude, and date range), then enriches those locations with NASA POWER data in the next step. This flow is handled via `enrich_locations`, which internally calls the common query handler before `await self._handle_enrich_locations(context, request, params)`.
 - Data availability: NASA POWER coverage is generally available for years >= 1981. Older dates are not available from NASA POWER and will be skipped/unavailable.

To use this agent, provide an artifact local_id with location records containing latitude, longitude, and date information. The agent will extract location data from the artifact and enrich it with NASA POWER weather and climate data.
"""


class BatchEnrichParams(BaseModel):

    locations_artifact: Optional[Artifact] = Field(default=None)
    address: Optional[str] = Field(default=None)
    weather_parameters: List[str] = Field(default=['T2M'])
    day: Optional[List[str]] = Field(default=None)
    month: Optional[List[str]] = Field(default=None)
    year: Optional[List[str]] = Field(default=None)
    start_range_date: Optional[str] = Field(default=None)
    end_range_date: Optional[str] = Field(default=None)
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
        
        async with context.begin_process(summary="Retrieving available parameters from NASA POWER API metadata") as process:
            process: IChatBioAgentProcess
            
            await process.log("Retrieving available parameters from NASA POWER API metadata")
            
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
            
            # await context.reply(response)

    async def _handle_common_query(self,
        context: ResponseContext,
        request: str,
        params: BatchEnrichParams,
        process: IChatBioAgentProcess) -> Optional[List[dict]]:
        
        await process.log("Creating locations JSON by extracting from request")
        await process.log(
            f"Data source: {params.source} (merra2 is the default data source.)"
        )
        await process.log(f"Common Query Params: {params}")
        
        address_to_use = params.address
        if address_to_use:
            await process.log(f"Geocoding address: {address_to_use}")
            geocoded_results = geocode_address(address_to_use)
            if geocoded_results:
                if len(geocoded_results) > 1:
                    results_message = f"Found {len(geocoded_results)} matching locations for '{address_to_use}':\n\n"
                    for idx, result in enumerate(geocoded_results, 1):
                        display_name = result.get('display_name', 'Unknown location')
                        lat = result.get('latitude')
                        lon = result.get('longitude')
                        results_message += f"{idx}. {display_name}\n   Coordinates: ({lat}, {lon})\n\n"
                    results_message += f"Using the first result by default. If you need a different location, please specify more details."
                    await context.reply(results_message)
                
                # Use the first result as default
                first_result = geocoded_results[0]
                latitude = first_result.get('latitude')
                longitude = first_result.get('longitude')
                display_name = first_result.get('display_name', address_to_use)
                await process.log(f"Selected location: {display_name}")
                await process.log(f"Using coordinates: ({latitude}, {longitude})")
            else:
                await process.log(f"Failed to geocode address '{address_to_use}'")
                await context.reply(
                    f"**Location not found.** Could not find coordinates for \"{address_to_use}\". "
                    "Please provide a valid address or specify coordinates (e.g. latitude, longitude)."
                )
                return
        else:
            await process.log("No address provided; will try to extract coordinates from request")
            await context.reply("No address provided; will try to extract coordinates from request")
            return

        if params.start_range_date and params.end_range_date:
            return [{
                    "decimalLatitude": float(latitude),
                    "decimalLongitude": float(longitude),
                    "startDate": params.start_range_date,
                    "endDate": params.end_range_date
                }]
            # artifact = await process.create_artifact(
            #     mimetype="application/json",
            #     description="Locations from address and start/end range date",
            #     content=json.dumps([                    {
            #         "decimalLatitude": float(latitude),
            #         "decimalLongitude": float(longitude),
            #         "startDate": params.start_range_date,
            #         "endDate": params.end_range_date
            #     }]).encode("utf-8"),
            #     metadata={
            #         "format": "json"
            #     },
            # )
            # await process.log("Created locations artifact")
            # return artifact

        days = []
        months = []
        years = []
        try:
            if params.day:
                days = sorted({int(d) for d in params.day})
            if params.month:
                months = sorted({int(m) for m in params.month})
            if params.year:
                years = sorted({int(y) for y in params.year})
        except ValueError:
            await process.log("Invalid year/month/day values in params; expected integers as strings.")
            await context.reply("Error: `year`, `month`, and `day` parameters must be integer strings, e.g. year=['2020'], month=['3'], day=['1','2','3'].")
            return None

        locations: list[dict] = []

        if len(months) == 0:
            await process.log(
                "Common query cannot proceed: `month` is empty. Months tie to seasons, so without a month "
                "we cannot choose meaningful dates or climate context."
            )
            await context.reply(
                "Error: Provide at least one `month` (e.g. month=['1','3']). Each month corresponds to a "
                "season; without a month we cannot infer dates or season-related values for you."
            )
            return None

        if len(years) == 0:
            cy = datetime.now(UTC).year
            years = [cy - 2, cy - 1, cy]
            await process.log(
                f"No `year` in params (user may not have a specific year in mind); "
                f"using the latest three calendar years {years} (UTC)."
            )

        for y in years:
            for m in months:
                if len(days) > 0:
                    for d in days:
                        try:
                            date_str = datetime(y, m, d).strftime("%Y-%m-%d")
                        except ValueError:
                            await process.log(f"Invalid calendar date: year={y}, month={m}, day={d}")
                            continue
                        locations.append(
                            {
                                "decimalLatitude": float(latitude),
                                "decimalLongitude": float(longitude),
                                "startDate": date_str,
                                "endDate": date_str
                            }
                        )
                else:
                    try:
                        start_str = datetime(y, m, 1).strftime("%Y-%m-%d")
                        if m == 12:
                            last_day = 31
                        else:
                            last_day = (datetime(y, m + 1, 1) - timedelta(days=1)).day
                        end_str = datetime(y, m, last_day).strftime("%Y-%m-%d")
                    except ValueError:
                        await process.log(f"Invalid year/month: year={y}, month={m}")
                        continue
                    locations.append(
                        {
                            "decimalLatitude": float(latitude),
                            "decimalLongitude": float(longitude),
                            "startDate": start_str,
                            "endDate": end_str
                        }
                    )
        # for location in locations:
        return locations
        
        # artifact = await process.create_artifact(
        #     mimetype="application/json",
        #     description="Locations from address and year/month/day",
        #     content=json.dumps(locations).encode("utf-8"),
        #     metadata={
        #         "format": "json",
        #         "source": "common_query",
        #         "record_count": len(locations),
        #     },
        # )
        # await process.log(f"Created locations artifact with {json.dumps(artifact)}")
        # return artifact

    async def _handle_enrich_locations(self, context: ResponseContext, request: str, params: Optional[BatchEnrichParams]):
        """Handle batch enrichment of location records with NASA POWER data"""
        source_content: dict | list = []
        generated_locations: Optional[list[dict]] = None

        if params.locations_artifact is None:
            async with context.begin_process(summary="Creating locations JSON by extracting from request") as process:
                process: IChatBioAgentProcess
                raw_locations = await self._handle_common_query(context, request, params, process)
                if raw_locations is None:
                    return
                generated_locations = raw_locations
                source_content = raw_locations

                artifact = await process.create_artifact(
                    mimetype="application/json",
                    description="Locations from address and year/month/day",
                    content=json.dumps(raw_locations).encode("utf-8"),
                    metadata={
                        "format": "json",
                        "source": "common_query",
                        "record_count": len(raw_locations),
                    },
                )
                await process.log("Created locations artifact")
                params.locations_artifact = artifact

        async with context.begin_process(summary="Enriching records with NASA POWER data") as process:
            process: IChatBioAgentProcess

            await process.log("Enriching records with NASA POWER data")
            
            if params is None:
                await context.reply("Error: No parameters provided. Please provide locations data.")
                return
            
            locations = None

            # await process.log(f"Enrich Params: {params}")

            try:
                if generated_locations is not None:
                    await process.log("Using generated locations from request parameters")
                    source_content = generated_locations
                elif params.locations_artifact is not None:
                    source_content = await retrieve_artifact_content(params.locations_artifact, process)
                else:
                    await context.reply(
                        "Error: No locations were provided. Please provide `address` or `locations_artifact`."
                    )
                    return

                if (
                    isinstance(source_content, list)
                    and source_content
                    and isinstance(source_content[0], dict)
                    and "decimalLatitude" in source_content[0]
                    and "decimalLongitude" in source_content[0]
                    and ("startDate" in source_content[0] or "endDate" in source_content[0])
                ):
                    await process.log("Using provided locations with startDate/endDate fields directly")
                    locations = [
                        {
                            "decimalLatitude": float(r["decimalLatitude"]),
                            "decimalLongitude": float(r["decimalLongitude"]),
                            "startDate": r["startDate"],
                            "endDate": r.get("endDate", r["startDate"])
                        }
                        for r in source_content
                    ]
                    locations = await sanitize_locations(locations, process)
                else:
                    schema = extract_json_schema(source_content)
                    await process.log(f"Extracted JSON schema from artifact content")

                    points = await extract_map_data_from_json(
                        request=request,
                        source_content=source_content,
                        schema=schema,
                        artifact=params.locations_artifact,
                        process=process,
                    )
                    if points is None:
                        await context.reply(
                            "Error: Could not extract geographic points and dates from the artifact "
                            "(JQ/LLM extraction failed). Ensure records include coordinates and dates "
                            "(e.g. startDate/endDate or eventDate), or verify ICHATBIO_JQ_MODEL / API access."
                        )
                        return

                    locations = [
                        {
                            "decimalLatitude": float(p.latitude),
                            "decimalLongitude": float(p.longitude),
                            "startDate": (p.startDate or p.date),
                            "endDate": (p.endDate or p.date)
                        }
                        for p in points
                    ]
                    print(f"Locations: {locations}")

                    locations = await sanitize_locations(locations, process)

                    # await process.log(
                    #     f"Extracted {len(locations)} location record(s) using JQ-based extraction"
                    # )
                        
            except ValueError:
                await context.reply("Error: Failed to retrieve the locations artifact content.")
                return

            if not isinstance(locations, list):
                await context.reply("Error: Location data must be a JSON array of location records")
                return
            
            if not locations:
                await context.reply("Error: Locations array is empty")
                return
            
            weather_parameters = params.weather_parameters
            frequency = params.frequency
            if frequency == 'monthly':
                frequency = 'daily'
            
            try:
                await process.log(f"Enriching locations with NASA POWER data...")

                enriched_locations = await enrich_locations_with_nasa_data(
                    locations=locations,
                    parameters=weather_parameters,
                    frequency=frequency,
                    source=params.source,
                    temporal=params.temporal,
                    time_standard=params.time,
                    process=process,
                )
                valid_date_rows = _count_valid_nasa_date_rows(enriched_locations)
                await process.log(
                    f"Enriched {len(enriched_locations)} location record(s) with NASA POWER data "
                    f"Successfully get NASA POWER data for {valid_date_rows} date/value rows",
                    data={
                        "source": [params.source] if params.source else [],
                        "parameters": weather_parameters,
                        "frequency": frequency,
                        "total_enriched_records": len(enriched_locations),
                        "valid_date_value_rows": valid_date_rows,
                    }
                )
                
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Location records enriched with NASA POWER data",
                        content=json.dumps(enriched_locations).encode("utf-8"),
                        metadata={
                            "format": "json",
                            "source": "NASA POWER",
                            "parameters": weather_parameters,
                            "frequency": frequency,
                            "total_records": len(enriched_locations),
                        }
                    )
                    await process.log(f"Created artifact with {len(enriched_locations)} location record(s)")
                except Exception as e:
                    await process.log(f"Warning: Failed to create artifact: {str(e)}")

                print(f"Enriched locations: {json.dumps(enriched_locations, indent=4)}")
                
                summary = (
                    f"**Batch Enrichment Complete**\n\n"
                    f"**Total Records from artifact:** {_count_artifact_records(source_content)}\n"
                    f"**Enriched locations (Valid records):** {len(enriched_locations)} location records\n"
                    f"**Successful get NASA POWER data (Valid values):** {valid_date_rows} date/value records\n"
                    f"**Parameters:** {', '.join(weather_parameters)}\n"
                    f"**Source:** {params.source}\n"
                    f"**Frequency:** {frequency}\n"
                )

                if enriched_locations:
                    summary += "**Sample Enriched Record:**\n"
                    sample = enriched_locations[0]
                    summary += f"  - Start Date: {sample.get('startDate')}\n"
                    summary += f"  - End Date: {sample.get('endDate')}\n"
                    summary += f"  - Location: ({sample.get('decimalLatitude')}, {sample.get('decimalLongitude')})\n"
                    props = sample.get("nasaPowerProperties")
                    if props:
                        nasa_data = props[0] if isinstance(props, list) else props
                        if isinstance(nasa_data, dict):
                            series = nasa_data.get("data")
                            if isinstance(series, list) and len(series) > 0:
                                first_data = series[0]
                                if isinstance(first_data, dict):
                                    raw_val = first_data.get("value")
                                    if raw_val is not None and isinstance(raw_val, (int, float)):
                                        value_str = f"{raw_val:.2f}"
                                    else:
                                        value_str = "null"
                                    summary += (
                                        f"  - {nasa_data.get('parameter')}: "
                                        f"{first_data.get('date')} = {value_str}\n"
                                    )
                
                summary += "\nArtifact with enriched location records is available."
                
                await context.reply(summary)
                
            except Exception as e:
                await process.log(f"Error during batch enrichment: {str(e)}")
                await context.reply(f"Error during batch enrichment: {str(e)}")


def create_app() -> Starlette:
    agent = NASAPowerAgent()
    app = build_agent_app(agent)
    return app
