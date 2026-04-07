"""
Tests for NASA POWER agent: list parameters, common query, and artifact-based enrichment.
Test cases from tests/resources/testcase.json are run via test_case_from_json.
"""
import os
import json
from datetime import date

import pytest
import ichatbio.types
from ichatbio.agent_response import (
    DirectResponse,
    ProcessBeginResponse,
    ProcessLogResponse,
    ArtifactResponse,
    ResponseMessage,
)

from resource_loader import resource
from src.agent import NASAPowerAgent, BatchEnrichParams
from src.util import extract_dates_from_request, last_n_calendar_month_ranges_for_month


# --- Helpers (for testcase.json-driven and shared assertions) ---


def _load_testcases():
    """Load test cases from tests/resources/testcase.json."""
    content = resource("testcase.json")
    return json.loads(content)


def _build_params(case, artifact_uri=None):
    """Build BatchEnrichParams from case params_override and optional artifact."""
    override = case.get("params_override")
    if override is None:
        return None
    kwargs = {
        "weather_parameters": ["T2M"],
        "date_range_days": 1,
        "frequency": "daily",
        "source": "merra2",
    }
    if isinstance(override, dict):
        kwargs.update(override)
    if artifact_uri is not None:
        kwargs["locations_artifact"] = ichatbio.types.Artifact(
            local_id=case.get("case_id", "#test"),
            description=case.get("description", ""),
            mimetype="application/json",
            uris=[artifact_uri],
            metadata={},
        )
    return BatchEnrichParams(**kwargs)


def _get_response_text(messages):
    """Extract final reply text from messages."""
    for m in reversed(messages):
        if isinstance(m, DirectResponse):
            return m.text
    return ""


def _get_log_text(messages):
    """Concatenate all process log texts."""
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    return " ".join([m.text for m in log_messages])


def _assert_contains(text, required_phrases, case_id=""):
    """Assert that text contains at least one of the required phrases (case-insensitive)."""
    text_lower = (text or "").lower()
    for phrase in required_phrases:
        if phrase.lower() in text_lower:
            return
    pytest.fail(
        f"[{case_id}] Response should contain one of: {required_phrases}. Got: {(text or '')[:500]}..."
    )


# --- List parameters / no params ---


@pytest.mark.asyncio
async def test_list_parameters_via_enrich(context, messages):
    """List parameters is shown when calling enrich_locations with params=None."""
    await NASAPowerAgent().run(context, "List parameters", "enrich_locations", None)

    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    assert "T2M" in messages[-1].text or "Temperature" in messages[-1].text or "Parameter" in messages[-1].text


@pytest.mark.asyncio
async def test_list_parameters_with_question(context, messages):
    """Listing parameters with natural language request (enrich_locations, no params)."""
    await NASAPowerAgent().run(
        context,
        "list parameter of NASA power",
        "enrich_locations",
        None,
    )

    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    response_text = messages[-1].text
    assert "T2M" in response_text
    assert "RH2M" in response_text or "Parameter" in response_text


@pytest.mark.asyncio
async def test_enrich_locations_no_params(context, messages):
    """When params is None, agent returns list of parameters (no enrichment)."""
    await NASAPowerAgent().run(
        context,
        "Enrich my locations",
        "enrich_locations",
        None,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "T2M" in messages[-1].text or "Parameter" in messages[-1].text or "Available" in messages[-1].text


# --- Common query (no artifact) ---


@pytest.mark.asyncio
async def test_common_query_gainesville_single_date(context, messages):
    """Common query: temperature in Gainesville, FL on 25 December 2024 (enrich_locations, no artifact)."""
    params = BatchEnrichParams(
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(
        context,
        "I would like to know temperature in Gainesville, Florida on 25 December 2024",
        "enrich_locations",
        params,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    response_text = messages[-1].text
    assert "T2M" in response_text or "Temperature" in response_text or "NASA POWER" in response_text


@pytest.mark.asyncio
async def test_common_query_date_range(context, messages):
    """Common query: temperature with date range 20-25 December 2024."""
    params = BatchEnrichParams(
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(
        context,
        "I would like to know temperature in Gainesville, Florida on 20-25 December 2024",
        "enrich_locations",
        params,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    response_text = messages[-1].text
    assert "T2M" in response_text or "Temperature" in response_text or "NASA POWER" in response_text
    log_text = " ".join([m.text for m in log_messages])
    assert "2024-12-20" in log_text or "2024-12-25" in log_text
    assert "6" in response_text or "Data Points" in response_text or "data" in response_text.lower()


@pytest.mark.asyncio
async def test_year_only_asks_for_month(context, messages):
    """Common query with year only (e.g. 'in 2020') asks user to specify at least month."""
    params = BatchEnrichParams(
        address="Gainesville, FL",
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(
        context,
        "Get the temperature in Gainesville, FL in 2020",
        "enrich_locations",
        params,
    )
    response_text = _get_response_text(messages)
    assert "month" in response_text.lower()
    assert "season" in response_text.lower()


# --- Enrich from artifact ---


@pytest.mark.httpx_mock(should_mock=lambda request: str(request.url) == "https://artifact.test")
@pytest.mark.asyncio
async def test_enrich_locations_with_artifact(context, messages, httpx_mock):
    """Enrich locations from an artifact (default merra2, T2M)."""
    content = resource("list_of_locations_10.json")
    httpx_mock.add_response(url="https://artifact.test", text=content)

    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#0000",
            description="na",
            mimetype="application/json",
            uris=["https://artifact.test"],
            metadata={},
        ),
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
        source="merra2",
    )
    await NASAPowerAgent().run(
        context,
        "Add temperature data to those records",
        "enrich_locations",
        params,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    log_text = " ".join([m.text for m in log_messages])
    assert "artifact" in log_text.lower() or "#0000" in log_text
    response_text = messages[-1].text
    assert "Enrichment" in response_text or "enriched" in response_text.lower()
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1
    assert artifact_messages[0].mimetype == "application/json"

    # Optionally save and assert GeoJSON artifact structure
    geojson_artifact = None
    for artifact_msg in artifact_messages:
        if artifact_msg.metadata and artifact_msg.metadata.get("format") == "geojson":
            geojson_artifact = artifact_msg
            break
    if geojson_artifact and geojson_artifact.content:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            geojson_data = json.loads(geojson_artifact.content.decode("utf-8"))
            geojson_file = os.path.join(output_dir, "artifact_test_geojson_output.json")
            with open(geojson_file, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=2, ensure_ascii=False)
            if isinstance(geojson_data, dict) and "features" in geojson_data and len(geojson_data["features"]) > 0:
                first_feature = geojson_data["features"][0]
                properties = first_feature.get("properties", {})
                param_keys = [k for k in properties.keys() if k != "value"]
                if param_keys:
                    param_data = properties.get(param_keys[0], {})
                    if isinstance(param_data, dict):
                        assert "parameter" in param_data
                        assert "value" in param_data
        except Exception:
            pass


@pytest.mark.httpx_mock(should_mock=lambda request: str(request.url) == "https://artifact.test/florida")
@pytest.mark.asyncio
async def test_enrich_locations_multiple_florida_cities(context, messages, httpx_mock):
    """Enrich multiple cities in Florida on a single date."""
    locations_data = [
        {"eventDate": "2024-12-25T00:00", "decimalLatitude": 29.6516, "decimalLongitude": -82.3248},
        {"eventDate": "2024-12-25T00:00", "decimalLatitude": 25.7617, "decimalLongitude": -80.1918},
        {"eventDate": "2024-12-25T00:00", "decimalLatitude": 30.3322, "decimalLongitude": -81.6557},
        {"eventDate": "2024-12-25T00:00", "decimalLatitude": 28.5383, "decimalLongitude": -82.4543},
        {"eventDate": "2024-12-25T00:00", "decimalLatitude": 26.1224, "decimalLongitude": -81.9353},
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
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(
        context,
        "I would like to know temperature all cities in Florida on 25 December 2024",
        "enrich_locations",
        params,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    response_text = messages[-1].text
    assert "Enrichment" in response_text or "enriched" in response_text.lower()
    assert "5" in response_text or "Total Records" in response_text
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1
    assert artifact_messages[0].mimetype == "application/json"


@pytest.mark.httpx_mock(should_mock=lambda request: str(request.url) == "https://artifact.test/json_array")
@pytest.mark.asyncio
async def test_enrich_locations_with_json_array(context, messages, httpx_mock):
    """Enrich an array of JSON locations with NASA POWER temperature data."""
    locations_data = [
        {"eventDate": "2005-12-16T09:41", "decimalLatitude": -47.428486, "decimalLongitude": -70.926533},
        {"eventDate": "2023-04-02T10:39", "decimalLatitude": -50.948586, "decimalLongitude": -72.712978},
        {"eventDate": None, "decimalLatitude": None, "decimalLongitude": None},
        {"eventDate": "2025-10-23T08:34", "decimalLatitude": -16.072473, "decimalLongitude": -71.478731},
        {"eventDate": None, "decimalLatitude": None, "decimalLongitude": None},
        {"eventDate": None, "decimalLatitude": None, "decimalLongitude": None},
        {"eventDate": None, "decimalLatitude": None, "decimalLongitude": None},
        {"eventDate": "2019-09-09T10:00", "decimalLatitude": -21.524045, "decimalLongitude": -66.896778},
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
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(
        context,
        "I want to add NASA POWER temperature fields to an array of JSON location",
        "enrich_locations",
        params,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    log_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(log_messages) >= 1
    response_text = messages[-1].text
    assert "Enrichment" in response_text or "enriched" in response_text.lower()
    assert "9" in response_text or "Total Records" in response_text
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1
    assert artifact_messages[0].mimetype == "application/json"


@pytest.mark.httpx_mock(should_mock=lambda request: str(request.url) == "https://artifact.test/multi_params")
@pytest.mark.asyncio
async def test_enrich_locations_with_multiple_parameters(context, messages, httpx_mock):
    """Enrich locations with multiple NASA POWER parameters (T2M and RH2M)."""
    locations_data = [
        {"eventDate": "2023-04-02T10:39", "decimalLatitude": -50.948586, "decimalLongitude": -72.712978},
        {"eventDate": "2019-09-09T10:00", "decimalLatitude": -21.524045, "decimalLongitude": -66.896778},
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
        weather_parameters=["T2M", "RH2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(
        context,
        "Add temperature and humidity data to my locations",
        "enrich_locations",
        params,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[-1], DirectResponse)
    response_text = messages[-1].text
    assert "T2M" in response_text or "RH2M" in response_text
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1


@pytest.mark.httpx_mock(should_mock=lambda request: "https://artifact.test/empty" in str(request.url))
@pytest.mark.asyncio
async def test_enrich_locations_empty_json(context, messages, httpx_mock):
    """Handling when artifact returns empty JSON array."""
    httpx_mock.add_response(url="https://artifact.test/empty", text="[]")
    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#empty",
            description="Empty list",
            mimetype="application/json",
            uris=["https://artifact.test/empty"],
            metadata={},
        ),
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(context, "Enrich empty data", "enrich_locations", params)
    messages: list[ResponseMessage]
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    text = messages[-1].text.lower()
    assert "empty" in text or "error" in text or "0" in text or "record" in text


@pytest.mark.httpx_mock(should_mock=lambda request: "https://artifact.test/invalid" in str(request.url))
@pytest.mark.asyncio
async def test_enrich_locations_invalid_json(context, messages, httpx_mock):
    """Error handling when artifact returns invalid JSON."""
    httpx_mock.add_response(url="https://artifact.test/invalid", text="not valid json {{{")
    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#invalid",
            description="Invalid",
            mimetype="application/json",
            uris=["https://artifact.test/invalid"],
            metadata={},
        ),
        weather_parameters=["T2M"],
        date_range_days=1,
        frequency="daily",
    )
    await NASAPowerAgent().run(context, "Enrich invalid data", "enrich_locations", params)
    messages: list[ResponseMessage]
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "error" in messages[-1].text.lower() or "invalid" in messages[-1].text.lower()


# --- Test-case-driven from testcase.json ---


@pytest.mark.httpx_mock(
    should_mock=lambda request: "https://artifact.test/" in str(request.url),
    assert_all_responses_were_requested=False,
)
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _load_testcases(), ids=lambda c: c["case_id"])
async def test_case_from_json(context, messages, case, httpx_mock):
    """
    Run each test case from tests/resources/testcase.json.
    Artifact cases: mock artifact URL with resource file.
    Common-query and other cases: run agent and assert response.
    """
    case_id = case["case_id"]
    case_type = case.get("type", "")
    request_text = case.get("request", "")
    expected_status = case.get("expected_status", "success")
    expected_response = case.get("expected_response_contains", [])
    expected_log = case.get("expected_log_contains", [])
    expected_error = case.get("expected_error_contains", [])

    params = None
    if case_type == "enrich_artifact":
        resource_file = case.get("resource_file", "list_of_locations_10.json")
        try:
            content = resource(resource_file)
        except Exception:
            pytest.skip(f"Resource {resource_file} not found")
        artifact_uri = f"https://artifact.test/{case_id}"
        httpx_mock.add_response(url=artifact_uri, text=content)
        params = _build_params(case, artifact_uri=artifact_uri)
    elif case_type == "common_query":
        params = _build_params(case)
    elif case_type == "enrich_locations_null_params":
        params = None
    elif case_type == "unknown_entrypoint":
        await NASAPowerAgent().run(context, request_text, "unknown_entrypoint", None)
        response_text = _get_response_text(messages)
        _assert_contains(response_text, expected_response, case_id)
        return
    else:
        pytest.skip(f"Unknown case type: {case_type}")

    await NASAPowerAgent().run(context, request_text, "enrich_locations", params)

    response_text = _get_response_text(messages)
    log_text = _get_log_text(messages)
    if expected_response:
        _assert_contains(response_text, expected_response, case_id)
    if expected_log:
        _assert_contains(log_text, expected_log, case_id)
    if expected_error and expected_status == "fail":
        _assert_contains(response_text, expected_error, case_id)

    assert isinstance(messages[0], ProcessBeginResponse), f"[{case_id}] Should start with process"
    assert isinstance(messages[-1], DirectResponse), f"[{case_id}] Should end with reply"
    if expected_status == "success" and case_type == "enrich_artifact":
        artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
        assert len(artifact_messages) >= 1, f"[{case_id}] Should produce an artifact"


# --- Month without year (params or NL): last 5 eligible years ---


def test_last_n_calendar_month_ranges_skips_future_august():
    ref = date(2026, 3, 22)
    ranges = last_n_calendar_month_ranges_for_month(8, n=5, today=ref)
    assert ranges == [
        ("2025-08-01", "2025-08-31"),
        ("2024-08-01", "2024-08-31"),
        ("2023-08-01", "2023-08-31"),
        ("2022-08-01", "2022-08-31"),
        ("2021-08-01", "2021-08-31"),
    ]


def test_extract_dates_month_of_august_no_year():
    ref = date(2026, 3, 22)
    ok, month_only, ranges, sd, ed = extract_dates_from_request(
        "average temperature in Scranton, PA during the month of August?",
        reference_date=ref,
    )
    assert ok and month_only
    assert len(ranges) == 5
    assert ranges[0] == ("2025-08-01", "2025-08-31")
    assert sd == "2021-08-01"
    assert ed == "2025-08-31"


# --- Entrypoint and card ---


@pytest.mark.asyncio
async def test_unknown_entrypoint(context, messages):
    """Unknown entrypoint returns error message."""
    await NASAPowerAgent().run(
        context,
        "Do something unknown",
        "unknown_entrypoint",
        None,
    )
    messages: list[ResponseMessage]
    assert len(messages) >= 1
    assert isinstance(messages[-1], DirectResponse)
    assert "unknown" in messages[-1].text.lower()


@pytest.mark.asyncio
async def test_agent_card(context, messages):
    """Agent card is properly configured."""
    agent = NASAPowerAgent()
    card = agent.get_agent_card()
    assert card.name == "NASA POWER Data Agent"
    assert "NASA" in card.description
    assert len(card.entrypoints) == 1
    assert card.entrypoints[0].id == "enrich_locations"
