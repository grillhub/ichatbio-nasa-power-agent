import json

import dotenv
import instructor.retry
import pydantic
import pytest
from instructor import from_openai, AsyncInstructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import Self

from conftest import resource
from src.util import (
    extract_json_schema,
    make_validated_location_response_model,
    LocationPropertyPaths,
    select_location_properties,
)

dotenv.load_dotenv()


def test_extract_schema():
    """Test extracting schema from a simple list of locations"""
    content = resource("list_of_locations.json")

    schema = extract_json_schema(json.loads(content))

    assert schema == {
        "$schema": "http://json-schema.org/schema#",
        "items": {
            "properties": {
                "eventDate": {"type": "string"},
                "decimalLatitude": {"type": "number"},
                "decimalLongitude": {"type": "number"},
            },
            "type": "object",
        },
        "type": "array",
    }


def test_extract_schema_for_buried_data():
    """Test extracting schema from buried location data"""
    content = resource("buried_list_of_locations.json")

    schema = extract_json_schema(json.loads(content))

    assert schema == {
        "$schema": "http://json-schema.org/schema#",
        "properties": {
            "points": {
                "items": {
                    "properties": {
                        "eventDate": {"type": "string"},
                        "decimalLatitude": {"type": "number"},
                        "decimalLongitude": {"type": "number"},
                    },
                    "type": "object",
                },
                "type": "array",
            },
            "version": {"type": "integer"},
        },
        "type": "object",
    }


def test_extract_schema_for_list_of_buried_locations():
    """Test extracting schema from list of buried locations"""
    content = resource("list_of_buried_locations.json")

    schema = extract_json_schema(json.loads(content))

    assert schema == {
        "$schema": "http://json-schema.org/schema#",
        "items": {
            "properties": {
                "location": {
                    "properties": {
                        "eventDate": {"type": "string"},
                        "decimalLatitude": {"type": "number"},
                        "decimalLongitude": {"type": "number"},
                    },
                    "type": "object",
                }
            },
            "type": "object",
        },
        "type": "array",
    }


def test_extract_schema_from_list_of_location_strings():
    """Test extracting schema from locations with string coordinates"""
    content = resource("list_of_location_strings.json")

    schema = extract_json_schema(json.loads(content))

    assert schema == {
        "$schema": "http://json-schema.org/schema#",
        "items": {
            "properties": {
                "eventDate": {"type": "string"},
                "decimalLatitude": {"type": "string"},
                "decimalLongitude": {"type": "string"},
            },
            "type": "object",
        },
        "type": "array",
    }


def test_model():
    """Test path validation model for location properties"""
    content = resource("buried_list_of_locations.json")
    schema = extract_json_schema(json.loads(content))
    model = make_validated_location_response_model(schema)

    with pytest.raises(pydantic.ValidationError):
        model(response=LocationPropertyPaths(latitude=["haha"], longitude=["haha"], date=["haha"]))

    with pytest.raises(pydantic.ValidationError):
        model(response=LocationPropertyPaths(latitude=["points"], longitude=["points"], date=["points"]))

    with pytest.raises(pydantic.ValidationError):
        model(response=LocationPropertyPaths(latitude=["decimalLatitude"], longitude=["decimalLongitude"], date=["eventDate"]))

    model(response=LocationPropertyPaths(latitude=["points", "decimalLatitude"], longitude=["points", "decimalLongitude"], date=["points", "eventDate"]))


@pytest.mark.asyncio
async def test_choose_paths():
    """Test selecting location property paths using LLM"""
    content = resource("buried_list_of_locations.json")
    schema = extract_json_schema(json.loads(content))
    paths = await select_location_properties("Extract location data with event dates", schema)

    assert paths == LocationPropertyPaths(
        latitude=["points", "decimalLatitude"],
        longitude=["points", "decimalLongitude"],
        date=["points", "eventDate"],
    )

