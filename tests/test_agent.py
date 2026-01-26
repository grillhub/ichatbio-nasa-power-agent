import pytest
import json
import ichatbio.types
from ichatbio.agent_response import DirectResponse, ProcessBeginResponse, ProcessLogResponse, ArtifactResponse, \
    ResponseMessage

from conftest import resource
from src.agent import NASAPowerAgent, NASAPowerQueryParams, BatchEnrichParams


@pytest.mark.asyncio
async def test_list_parameters(context, messages):
    """Test the list_parameters entrypoint"""
    await NASAPowerAgent().run(context, "List parameters", "list_parameters", None)
    
    # Message objects are restricted to the following types:
    messages: list[ResponseMessage]
    
    # Should start with a process and end with a reply
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    # The response should contain parameter names
    assert "T2M" in messages[-1].text or "Temperature" in messages[-1].text


@pytest.mark.asyncio
async def test_list_parameters_with_question(context, messages):
    """Test listing parameters with natural language question"""
    await NASAPowerAgent().run(
        context, 
        "list parameter of NASA power", 
        "list_parameters", 
        None
    )
    
    messages: list[ResponseMessage]
    
    # Should start with process and end with reply
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    
    # Response should contain common parameters
    response_text = messages[-1].text
    assert "T2M" in response_text  # Temperature
    assert "RH2M" in response_text  # Relative Humidity


@pytest.mark.asyncio
async def test_query_weather_gainesville_florida(context, messages):
    """Test querying temperature in Gainesville, Florida on December 25, 2024"""
    params = NASAPowerQueryParams(
        latitude=29.6516,  # Gainesville, Florida coordinates
        longitude=-82.3248,
        parameter="T2M",
        start_date="2024-12-25",
        end_date="2024-12-25",
        frequency="daily"
    )
    
    await NASAPowerAgent().run(
        context,
        "I would like to know temperature in Gainesville, Florida on 25 December 2024",
        "query_weather",
        params
    )
    
    messages: list[ResponseMessage]
    
    # Should start with process, have logs, and end with reply
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    
    # Check for process logs
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    
    # Response should contain location and data info
    response_text = messages[-1].text
    assert "T2M" in response_text or "Temperature" in response_text
    
    # Check for artifact creation (JSON data)
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1


@pytest.mark.asyncio
async def test_query_weather_date_range(context, messages):
    """Test querying temperature with multiple date range (20-25 December 2024)"""
    params = NASAPowerQueryParams(
        latitude=29.6516,  # Gainesville, Florida coordinates
        longitude=-82.3248,
        parameter="T2M",
        start_date="2024-12-20",
        end_date="2024-12-25",
        frequency="daily"
    )
    
    await NASAPowerAgent().run(
        context,
        "I would like to know temperature in Gainesville, Florida on 20-25 December 2024",
        "query_weather",
        params
    )
    
    messages: list[ResponseMessage]
    
    # Should start with process, have logs, and end with reply
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    
    # Check for process logs
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    
    # Response should contain location and data info
    response_text = messages[-1].text
    assert "T2M" in response_text or "Temperature" in response_text
    
    # Should mention date range in logs
    log_text = " ".join([m.text for m in log_messages])
    assert "2024-12-20" in log_text or "2024-12-25" in log_text
    
    # Check for artifact creation (JSON data)
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1
    
    # Should have multiple data points (one for each day in range)
    # 6 days: Dec 20, 21, 22, 23, 24, 25
    assert "6" in response_text or "Data Points" in response_text


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test/florida"
)
@pytest.mark.asyncio
async def test_enrich_locations_multiple_florida_cities(context, messages, httpx_mock):
    """Test enriching multiple cities in Florida on a single date"""
    # Multiple cities in Florida on December 25, 2024
    locations_data = [
        {
            "eventDate": "2024-12-25T00:00",
            "decimalLatitude": 29.6516,  # Gainesville
            "decimalLongitude": -82.3248
        },
        {
            "eventDate": "2024-12-25T00:00",
            "decimalLatitude": 25.7617,  # Miami
            "decimalLongitude": -80.1918
        },
        {
            "eventDate": "2024-12-25T00:00",
            "decimalLatitude": 30.3322,  # Jacksonville
            "decimalLongitude": -81.6557
        },
        {
            "eventDate": "2024-12-25T00:00",
            "decimalLatitude": 28.5383,  # Tampa
            "decimalLongitude": -82.4543
        },
        {
            "eventDate": "2024-12-25T00:00",
            "decimalLatitude": 26.1224,  # Fort Myers
            "decimalLongitude": -81.9353
        }
    ]
    locations_json = json.dumps(locations_data)
    httpx_mock.add_response(url="https://artifact.test/florida", text=locations_json)
    
    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#0000",
            description="Florida cities locations",
            mimetype="application/json",
            uris=["https://artifact.test/florida"],
            metadata={},
        ),
        weather_parameters=['T2M'],
        date_range_days=1,
        frequency='daily'
    )
    
    await NASAPowerAgent().run(
        context,
        "I would like to know temperature all cities in Florida on 25 December 2024",
        "enrich_locations",
        params
    )
    
    messages: list[ResponseMessage]
    
    # Should start with process and end with reply
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    
    # Check for process logs
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    
    # Response should mention enrichment results
    response_text = messages[-1].text
    assert "Enrichment" in response_text or "enriched" in response_text.lower()
    assert "5" in response_text or "Total Records" in response_text  # 5 cities
    
    # Check for artifact creation (enriched JSON data)
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1
    
    # Verify the artifact contains the enriched data
    artifact = artifact_messages[0]
    assert artifact.mimetype == "application/json"


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test/json_array"
)
@pytest.mark.asyncio
async def test_enrich_locations_with_json_array(context, messages, httpx_mock):
    """Test enriching an array of JSON locations with NASA POWER temperature data"""
    
    # The JSON array provided by the user
    locations_data = [
        {
            "eventDate": "2005-12-16T09:41",
            "decimalLatitude": -47.428486,
            "decimalLongitude": -70.926533
        },
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
            "eventDate": "2025-10-23T08:34",
            "decimalLatitude": -16.072473,
            "decimalLongitude": -71.478731
        },
        {
            "eventDate": None,
            "decimalLatitude": None,
            "decimalLongitude": None
        },
        {
            "eventDate": None,
            "decimalLatitude": None,
            "decimalLongitude": None
        },
        {
            "eventDate": None,
            "decimalLatitude": None,
            "decimalLongitude": None
        },
        {
            "eventDate": None,
            "decimalLatitude": None,
            "decimalLongitude": None
        },
        {
            "eventDate": "2019-09-09T10:00",
            "decimalLatitude": -21.524045,
            "decimalLongitude": -66.896778
        }
    ]
    locations_json = json.dumps(locations_data)
    httpx_mock.add_response(url="https://artifact.test/json_array", text=locations_json)
    
    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#0001",
            description="JSON array locations",
            mimetype="application/json",
            uris=["https://artifact.test/json_array"],
            metadata={},
        ),
        weather_parameters=['T2M'],
        date_range_days=1,
        frequency='daily'
    )
    
    await NASAPowerAgent().run(
        context,
        "I want to add NASA POWER temperature fields to an array of JSON location",
        "enrich_locations",
        params
    )
    
    messages: list[ResponseMessage]
    
    # Should start with process and end with reply
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    
    # Check for process logs
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    
    # Response should mention enrichment results
    response_text = messages[-1].text
    assert "Enrichment" in response_text or "enriched" in response_text.lower()
    assert "9" in response_text or "Total Records" in response_text  # 9 total records
    
    # Check for artifact creation (enriched JSON data)
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1
    
    # Verify the artifact contains the enriched data
    artifact = artifact_messages[0]
    assert artifact.mimetype == "application/json"


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test/multi_params"
)
@pytest.mark.asyncio
async def test_enrich_locations_with_multiple_parameters(context, messages, httpx_mock):
    """Test enriching locations with multiple NASA POWER parameters (T2M and RH2M)"""
    
    locations_data = [
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
    locations_json = json.dumps(locations_data)
    httpx_mock.add_response(url="https://artifact.test/multi_params", text=locations_json)
    
    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#0002",
            description="Multi-parameter locations",
            mimetype="application/json",
            uris=["https://artifact.test/multi_params"],
            metadata={},
        ),
        weather_parameters=['T2M', 'RH2M'],  # Temperature and Humidity
        date_range_days=1,
        frequency='daily'
    )
    
    await NASAPowerAgent().run(
        context,
        "Add temperature and humidity data to my locations",
        "enrich_locations",
        params
    )
    
    messages: list[ResponseMessage]
    
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    
    response_text = messages[-1].text
    assert "T2M" in response_text or "RH2M" in response_text
    
    # Check for artifact
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1


@pytest.mark.asyncio
async def test_enrich_locations_empty_json(context, messages):
    """Test error handling when empty JSON array is provided"""
    
    params = BatchEnrichParams(
        locations_json="[]",
        weather_parameters=['T2M'],
        date_range_days=1,
        frequency='daily'
    )
    
    await NASAPowerAgent().run(
        context,
        "Enrich empty data",
        "enrich_locations",
        params
    )
    
    messages: list[ResponseMessage]
    
    # Should return an error message
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "empty" in messages[-1].text.lower() or "error" in messages[-1].text.lower()


@pytest.mark.asyncio
async def test_enrich_locations_invalid_json(context, messages):
    """Test error handling when invalid JSON is provided"""
    
    params = BatchEnrichParams(
        locations_json="not valid json {{{",
        weather_parameters=['T2M'],
        date_range_days=1,
        frequency='daily'
    )
    
    await NASAPowerAgent().run(
        context,
        "Enrich invalid data",
        "enrich_locations",
        params
    )
    
    messages: list[ResponseMessage]
    
    # Should return an error message
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "error" in messages[-1].text.lower() or "invalid" in messages[-1].text.lower()


@pytest.mark.asyncio
async def test_enrich_locations_no_params(context, messages):
    """Test error handling when no parameters are provided"""
    
    await NASAPowerAgent().run(
        context,
        "Enrich my locations",
        "enrich_locations",
        None
    )
    
    messages: list[ResponseMessage]
    
    # Should return an error message about missing data
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "error" in messages[-1].text.lower() or "no" in messages[-1].text.lower()


@pytest.mark.asyncio
async def test_unknown_entrypoint(context, messages):
    """Test handling of unknown entrypoint"""
    
    await NASAPowerAgent().run(
        context,
        "Do something unknown",
        "unknown_entrypoint",
        None
    )
    
    messages: list[ResponseMessage]
    
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "unknown" in messages[-1].text.lower()

@pytest.mark.asyncio
async def test_agent_card(context, messages):
    """Test that agent card is properly configured"""
    agent = NASAPowerAgent()
    card = agent.get_agent_card()
    
    assert card.name == "NASA POWER Data Agent"
    assert "NASA" in card.description
    assert len(card.entrypoints) == 3
    
    # Check entrypoint IDs
    entrypoint_ids = [ep.id for ep in card.entrypoints]
    assert "query_weather" in entrypoint_ids
    assert "list_parameters" in entrypoint_ids
    assert "enrich_locations" in entrypoint_ids
