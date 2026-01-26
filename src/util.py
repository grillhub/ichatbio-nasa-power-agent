import json
import re
from typing import Optional, Self, Iterator, Union, Literal
import httpx
from genson import SchemaBuilder
from genson.schema.strategies import Object
from ichatbio.agent_response import IChatBioAgentProcess
from ichatbio.types import Artifact
from instructor import from_openai, retry, AsyncInstructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

JSON = dict | list | str | int | float | None
"""JSON-serializable primitive types that work with functions like json.dumps(). Note that dicts and lists may contain
content that is not JSON-serializable."""

Path = list[str]


def contains_non_null_content(content: JSON):
    """
    Returns True only if the JSON-serializable content contains a non-empty value. For example, returns True for [[1]]
    and False for [[]].
    """
    match content:
        case None:
            return False
        case list() as l:
            return any([contains_non_null_content(v) for v in l])
        case dict() as d:
            return any([contains_non_null_content(v) for k, v in d.items()])
        case _:
            return True


# JSON schema extraction


class NoRequiredObject(Object):
    KEYWORDS = tuple(kw for kw in Object.KEYWORDS if kw != "required")

    # Remove "required" from the output if present
    def to_schema(self):
        schema = super().to_schema()
        if "required" in schema:
            del schema["required"]
        return schema


class NoRequiredSchemaBuilder(SchemaBuilder):
    """SchemaBuilder that does not use the "required" keyword, which roughly doubles the length of the schema string,
    and also isn't very helpful for our purposes."""

    EXTRA_STRATEGIES = (NoRequiredObject,)


def extract_json_schema(content: str) -> dict:
    builder = NoRequiredSchemaBuilder()
    builder.add_object(content)
    schema = builder.to_schema()
    return schema


async def parse_locations_json(
    json_content: str, process: IChatBioAgentProcess
) -> list:
    json_content = json_content.strip()
    
    try:
        # Try parsing the JSON - handle potential double-encoding
        try:
            locations = json.loads(json_content)
        except json.JSONDecodeError as first_error:
            # If first parse fails with "Extra data", try to find where valid JSON ends
            if "Extra data" in str(first_error):
                # Try to parse just the first valid JSON value
                # This handles cases where multiple JSON values are concatenated
                try:
                    # Find the position where the error occurred
                    error_msg = str(first_error)
                    if "char" in error_msg:
                        # Extract the character position
                        match = re.search(r'char (\d+)', error_msg)
                        if match:
                            pos = int(match.group(1))
                            # Try parsing just up to that position
                            partial_json = json_content[:pos]
                            locations = json.loads(partial_json)
                            await process.log(
                                f"Warning: Extra data found after JSON. Parsed {len(locations)} records from partial JSON."
                            )
                        else:
                            raise first_error
                    else:
                        raise first_error
                except (json.JSONDecodeError, ValueError):
                    raise first_error
            else:
                # For other JSON errors, try double-decoding
                try:
                    # If the content looks like it might be a JSON-encoded string, try decoding it
                    if json_content.startswith('"') and json_content.endswith('"'):
                        decoded = json.loads(json_content)
                        if isinstance(decoded, str):
                            locations = json.loads(decoded)
                        else:
                            raise first_error
                    else:
                        raise first_error
                except (json.JSONDecodeError, ValueError):
                    # If that also fails, raise the original error
                    raise first_error
    
    except json.JSONDecodeError as e:
        # Log a snippet of the content for debugging (first 200 chars)
        content_snippet = json_content[:200] if len(json_content) > 200 else json_content
        await process.log(f"JSON parsing failed. Content snippet: {content_snippet}...")
        raise
    
    if not isinstance(locations, list):
        raise ValueError("Location data must be a JSON array of location records")
    
    return locations


async def retrieve_artifact_content(
    artifact: Artifact, process: IChatBioAgentProcess
) -> JSON:
    async with httpx.AsyncClient(follow_redirects=True) as internet:
        for url in artifact.get_urls():
            await process.log(
                f"Retrieving artifact {artifact.local_id} content from {url}"
            )
            response = await internet.get(url)
            if response.is_success:
                return response.json()  # TODO: catch exception?
            else:
                await process.log(
                    f"Failed to retrieve artifact content: {response.reason_phrase} ({response.status_code})"
                )
                raise ValueError()
        else:
            await process.log("Failed to find artifact content")
            raise ValueError()


# Location property path selection


class GiveUp(BaseModel):
    reason: str


class LocationPropertyPaths(BaseModel):
    """Standard location format with separate latitude and longitude scalar fields."""
    format: Literal["scalar_fields"] = "scalar_fields"
    latitude: Path
    longitude: Path
    date: Path = Field(
        description="The property path to the event date or timestamp field (e.g., eventDate, date, timestamp)."
    )


class LocationStringPaths(BaseModel):
    """Location format where lat,lng are stored as a single string like '-27.45,151.93'."""
    format: Literal["location_string"] = "location_string"
    location_string: Path = Field(
        description="The property path to a string field containing 'latitude,longitude' (e.g., '-27.45,151.93')."
    )
    date: Path = Field(
        description="The property path to the event date or timestamp field."
    )


class GeoJSONCoordinatesPaths(BaseModel):
    """Location format where coordinates are stored as GeoJSON [longitude, latitude] array."""
    format: Literal["geojson_coordinates"] = "geojson_coordinates"
    coordinates: Path = Field(
        description="The property path to a GeoJSON coordinates array [longitude, latitude]."
    )
    date: Path = Field(
        description="The property path to the event date or timestamp field."
    )


# Union type for all location path formats
LocationPaths = LocationPropertyPaths | LocationStringPaths | GeoJSONCoordinatesPaths


def trace_path_in_schema(schema: dict, target_path: list[str], index=0):
    if index < len(target_path):
        match schema.get("type"):
            case "object":
                next_property = schema["properties"].get(target_path[index])
                if next_property:
                    return trace_path_in_schema(next_property, target_path, index + 1)
            case "array":
                return trace_path_in_schema(schema["items"], target_path, index)

    return target_path[:index], schema


def make_validated_location_response_model(schema: dict):
    def validate_scalar_path(path: list[str], allowed_types=("integer", "number", "string")):
        trace, terminal_schema = trace_path_in_schema(schema, path)

        if trace != path:
            terminal_type = terminal_schema.get("type", "unknown")
            raise ValueError(
                f'Path does not exist in provided schema. Tip: {terminal_type} at path {trace} does not contain a property named "{path[len(trace)]}"'
            )

        terminal_type = terminal_schema.get("type", "unknown")
        
        # Handle nullable types: if type is a list (e.g., ['null', 'number']), 
        # check if any non-null type is in the allowed types
        if isinstance(terminal_type, list):
            # Filter out 'null' and check if any remaining type is allowed
            non_null_types = [t for t in terminal_type if t != "null"]
            if not any(t in allowed_types for t in non_null_types):
                raise ValueError(
                    f'Path {trace} in the schema has invalid type "{terminal_type}"; expected {allowed_types}'
                )
        elif terminal_type not in allowed_types:
            raise ValueError(
                f'Path {trace} in the schema has invalid type "{terminal_type}"; expected {allowed_types}'
            )

        return path
    
    def validate_array_path(path: list[str]):
        trace, terminal_schema = trace_path_in_schema(schema, path)

        if trace != path:
            terminal_type = terminal_schema.get("type", "unknown")
            raise ValueError(
                f'Path does not exist in provided schema. Tip: {terminal_type} at path {trace} does not contain a property named "{path[len(trace)]}"'
            )

        if terminal_schema.get("type") != "array":
            terminal_type = terminal_schema.get("type", "unknown")
            raise ValueError(
                f'Path {trace} in the schema has invalid type "{terminal_type}"; expected "array"'
            )

        return path

    class ResponseModel(BaseModel):
        response: LocationPropertyPaths | LocationStringPaths | GeoJSONCoordinatesPaths | GiveUp

        @model_validator(mode="after")
        def validate(self) -> Self:
            match self.response:
                case LocationPropertyPaths() as paths:
                    validate_scalar_path(paths.latitude)
                    validate_scalar_path(paths.longitude)
                    validate_scalar_path(paths.date)
                case LocationStringPaths() as paths:
                    validate_scalar_path(paths.location_string)
                    validate_scalar_path(paths.date)
                case GeoJSONCoordinatesPaths() as paths:
                    validate_array_path(paths.coordinates)
                    validate_scalar_path(paths.date)
            return self

    return ResponseModel


SYSTEM_PROMPT = """\
Your task is to look at a JSON schema and map paths in the schema to location properties that the user is interested in, as 
defined by a provided data model.

A path is a list of property names that navigate through the JSON structure. For example, ["results", "data", "geo", "lat"].

Location data can be stored in THREE different formats. Choose the one that best matches the schema:

1. **format: "scalar_fields"** - Separate latitude and longitude scalar fields
   Use when there are distinct properties for lat and lng (e.g., "decimalLatitude", "decimalLongitude", "lat", "lon").
   Example paths:
   - latitude: ["results", "decimalLatitude"]
   - longitude: ["results", "decimalLongitude"]
   - date: ["results", "eventDate"]

2. **format: "location_string"** - Single string containing "latitude,longitude"
   Use when location is stored as a comma-separated string like "-27.45,151.93".
   Common property names: "location", "latlng", "coords"
   Example paths:
   - location_string: ["results", "location"]
   - date: ["results", "observed_on"]

3. **format: "geojson_coordinates"** - GeoJSON coordinates array [longitude, latitude]
   Use when coordinates are in a GeoJSON structure with type "Point" and coordinates array.
   Note: GeoJSON uses [longitude, latitude] order (opposite of typical lat/lng).
   Example paths:
   - coordinates: ["results", "geojson", "coordinates"]
   - date: ["results", "time_observed_at"]

IMPORTANT:
- For date, look for properties like: eventDate, date, timestamp, time_observed_at, observed_on, created_at
- Choose "location_string" format if there's a "location" property with type "string" containing lat,lng (e.g. iNaturalist)
- Choose "geojson_coordinates" format if there's a geojson.coordinates array (e.g. iNaturalist: ["results","geojson","coordinates"])
- Choose "scalar_fields" format only if there are separate scalar properties for latitude and longitude
- Do NOT use GiveUp when coordinates are in geojson.coordinates or in a location string—use geojson_coordinates or location_string instead
- If none of these formats match, give up with a clear reason
"""


async def select_location_properties(request: str, schema: dict):
    model = make_validated_location_response_model(schema)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the schema of my data:\n\n{schema}",
        },
        {"role": "user", "content": request},
    ]

    client: AsyncInstructor = from_openai(AsyncOpenAI())
    try:
        generation = await client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            response_model=model,
            messages=messages,
            max_retries=5,
        )
    except retry.InstructorRetryException as e:
        raise

    return generation.response


def read_path(content: JSON, path: list[str]) -> Iterator:
    # If path is exhausted, yield the content as-is (including arrays like coordinates)
    if len(path) == 0:
        yield content
        return
    
    match content:
        case list() as records:
            # Traverse into array of records (e.g., "results" array)
            for record in records:
                yield from read_path(record, path)
        case dict() as record:
            next_property = record.get(path[0])
            yield from read_path(next_property, path[1:])


def parse_location_string(location_str: str) -> tuple[float, float] | None:
    """
    Parse a location string in format "latitude,longitude" (e.g., "-27.45,151.93").
    Returns (latitude, longitude) tuple or None if parsing fails.
    """
    if not location_str or not isinstance(location_str, str):
        return None
    try:
        parts = location_str.split(",")
        if len(parts) != 2:
            return None
        lat = float(parts[0].strip())
        lng = float(parts[1].strip())
        return (lat, lng)
    except (ValueError, AttributeError):
        return None


def parse_geojson_coordinates(coordinates: list) -> tuple[float, float] | None:
    """
    Parse GeoJSON coordinates array [longitude, latitude].
    Note: GeoJSON uses [lng, lat] order, but we return (lat, lng).
    Returns (latitude, longitude) tuple or None if parsing fails.
    """
    if not coordinates or not isinstance(coordinates, list):
        return None
    try:
        if len(coordinates) < 2:
            return None
        lng = float(coordinates[0])
        lat = float(coordinates[1])
        return (lat, lng)
    except (ValueError, TypeError, IndexError):
        return None


def try_extract_locations_heuristic(content: JSON) -> list[dict] | None:
    """
    Try to extract location records from common JSON shapes (e.g. iNaturalist API)
    when schema-based extraction returns GiveUp.

    Looks for:
    - records array: top-level list, or object keys "results", "data", "records", "observations"
    - coordinates: record.geojson.coordinates [lon,lat], record.location "lat,lon", or record.decimalLatitude/decimalLongitude
    - date: observed_on, time_observed_at, eventDate, date, created_at

    Returns a list of {eventDate, decimalLatitude, decimalLongitude} or None if nothing could be extracted.
    """
    records: list = []
    if isinstance(content, list):
        records = content
    elif isinstance(content, dict):
        for key in ("results", "data", "records", "observations"):
            cand = content.get(key)
            if isinstance(cand, list) and len(cand) > 0:
                records = cand
                break

    if not records:
        return None

    date_keys = ("observed_on", "time_observed_at", "eventDate", "date", "created_at")
    out: list[dict] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        lat, lon = None, None

        # 1) geojson.coordinates [lon, lat]
        g = r.get("geojson")
        if isinstance(g, dict):
            parsed = parse_geojson_coordinates(g.get("coordinates"))
            if parsed:
                lat, lon = parsed

        # 2) location "lat,lon" (e.g. iNaturalist)
        if lat is None or lon is None:
            parsed = parse_location_string(r.get("location") if isinstance(r.get("location"), str) else None)
            if parsed:
                lat, lon = parsed

        # 3) scalar decimalLatitude, decimalLongitude
        if lat is None or lon is None:
            la, lo = r.get("decimalLatitude"), r.get("decimalLongitude")
            if la is not None and lo is not None:
                try:
                    lat, lon = float(la), float(lo)
                except (TypeError, ValueError):
                    pass

        # date: first string found in date_keys, normalized to YYYY-MM-DD
        date_val = None
        for k in date_keys:
            v = r.get(k)
            if v is not None and isinstance(v, str):
                date_val = v.split("T")[0] if "T" in v else v
                break

        if lat is not None and lon is not None and date_val is not None:
            out.append({"decimalLatitude": lat, "decimalLongitude": lon, "eventDate": date_val})

    return out if out else None
