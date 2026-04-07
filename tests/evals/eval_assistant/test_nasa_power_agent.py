import json
import pathlib

import instructor
import pytest
import yaml
import ichatbio.types
from ichatbio.agent_response import DirectResponse, ProcessLogResponse
from openai import AsyncOpenAI

from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from resource_loader import resource
from src.agent import NASAPowerAgent, BatchEnrichParams


file = pathlib.Path(__file__).parent / "test_sets" / "generate_expert_requests_nasa.yaml"
with open(file) as f:
    tests = yaml.safe_load(f)["test_cases"]

equivalence = GEval(
    name="Equivalence",
    criteria="Determine if the 'actual output' is semantically equivalent to 'expected output'. Cosmetic differences"
    " are okay.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model="gpt-4.1-mini",
)

def _get_response_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, DirectResponse):
            return m.text or ""
    return ""

def _get_log_text(messages) -> str:
    logs = [m.text for m in messages if isinstance(m, ProcessLogResponse) and m.text]
    return " ".join(logs)

async def _batch_enrich_params_from_user_message(user_message: str) -> BatchEnrichParams:
    client: instructor.AsyncInstructor = instructor.from_openai(AsyncOpenAI())
    return await client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_model=BatchEnrichParams,
        messages=[
            {
                "role": "system",
                "content": (
                    "You map a natural-language request (user's message) into BatchEnrichParams for the NASA POWER tool. "
                    "Read response_model to understand the fields and their constraints."
                    "Response should be in the format of response_model."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        max_retries=2,
    )


@pytest.mark.httpx_mock(should_mock=lambda request: str(request.url) == "https://artifact.test")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message,expected,artifact",
    [(t["user_message"], t["expected"], t["artifact"]) for t in tests],
)
async def test_nasa_power_agent_eval(context, messages, httpx_mock, user_message, expected, artifact):

    extracted = await _batch_enrich_params_from_user_message(user_message)
    extracted_without_artifact = extracted.model_dump(exclude={"locations_artifact"})

    artifact_url = "https://artifact.test"

    if artifact != "None":
        content = resource(artifact)
        httpx_mock.add_response(url=artifact_url, text=content)
        locations_artifact = ichatbio.types.Artifact(
            local_id="#eval",
            description="eval locations",
            mimetype="application/json",
            uris=[artifact_url],
            metadata={},
        )
        params = BatchEnrichParams(locations_artifact=locations_artifact, **extracted_without_artifact)
    else:
        probe = BatchEnrichParams(locations_artifact=None, **extracted_without_artifact)
        async with context.begin_process(summary="Creating locations JSON by extracting from request") as process:
            raw_locations = await NASAPowerAgent()._handle_common_query(
                context, user_message, probe, process
            )
        if raw_locations is None:
            params = probe
        else:
            httpx_mock.add_response(url=artifact_url, text=json.dumps(raw_locations))
            locations_artifact = ichatbio.types.Artifact(
                local_id="#eval",
                description="eval locations",
                mimetype="application/json",
                uris=[artifact_url],
                metadata={},
            )
            params = BatchEnrichParams(locations_artifact=locations_artifact, **extracted_without_artifact)

    await NASAPowerAgent().run(context, user_message, "enrich_locations", params)

    actual_response = _get_response_text(messages)
    actual_logs = _get_log_text(messages)
    actual_output = actual_response if actual_response else actual_logs

    test_case = LLMTestCase(
        input=user_message,
        expected_output=expected,
        actual_output=actual_output,
    )
    assert_test(test_case, [equivalence])