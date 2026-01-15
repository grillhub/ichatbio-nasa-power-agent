import json
import re
from typing import Optional, Self, Iterator
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
    latitude: Path
    longitude: Path
    date: Path = Field(
        description="The property path to the event date or timestamp field (e.g., eventDate, date, timestamp)."
    )


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


def make_validated_location_response_model(
    schema: dict, allowed_types=("integer", "number", "string")
):
    def validate_path(path: list[str]):
        trace, terminal_schema = trace_path_in_schema(schema, path)

        if trace != path:
            terminal_type = terminal_schema["type"]
            raise ValueError(
                f'Path does not exist in provided schema. Tip: {terminal_type} at path {trace} does not contain a property named "{path[len(trace)]}"'
            )

        if terminal_schema["type"] not in allowed_types:
            terminal_type = terminal_schema["type"]
            raise ValueError(
                f'Path {trace} in the schema has invalid type "{terminal_type}"; expected {allowed_types}'
            )

        return path

    class ResponseModel(BaseModel):
        response: GiveUp | LocationPropertyPaths

        @model_validator(mode="after")
        def validate(self) -> Self:
            match self.response:
                case LocationPropertyPaths() as paths:
                    validate_path(paths.latitude)
                    validate_path(paths.longitude)
                    validate_path(paths.date)
            return self

    return ResponseModel


SYSTEM_PROMPT = """\
Your task is to look at a JSON schema and map paths in the schema to location properties that the user is interested in, as 
defined by a provided data model.

A path is a list of property names that point to a scalar property. For example,

latitude: ["records", "data", "geo", "decimalLatitude"]
longitude: ["records", "data", "geo", "decimalLongitude"]
date: ["records", "data", "eventDate"]

You need to identify:
- latitude: A path to a numeric property containing latitude values (typically named latitude, decimalLatitude, lat, etc.)
- longitude: A path to a numeric property containing longitude values (typically named longitude, decimalLongitude, lon, lng, etc.)
- date: A path to a string property containing date/timestamp values (typically named eventDate, date, timestamp, time, etc.)
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
    match content:
        case list() as records:
            for record in records:
                yield from read_path(record, path)
        case dict() as record:
            next_property = record.get(path[0])
            yield from read_path(next_property, path[1:])
        case _ as scalar if len(path) == 0:
            yield scalar
