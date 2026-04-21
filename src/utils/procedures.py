import json
import os

import httpx
from genson import SchemaBuilder
from genson.schema.strategies import Object
from ichatbio.agent_response import IChatBioAgentProcess
from ichatbio.types import Artifact
from instructor.exceptions import InstructorRetryException

from .parse import _generate_and_run_jq_query, GiveUp, JQQuery
from .plot import LabeledGeoPoint
from .util import JSON


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


def extract_json_schema(content: JSON) -> dict:
    builder = NoRequiredSchemaBuilder()
    builder.add_object(content)
    schema = builder.to_schema()
    return schema


async def retrieve_artifact_content(
    artifact: Artifact, process: IChatBioAgentProcess
) -> JSON:
    async with httpx.AsyncClient(follow_redirects=True) as internet:
        for url in artifact.get_urls():
            await process.log(
                f"Retrieving artifact {artifact.local_id} content from {url}"
            )

            if "localhost" in url and os.getenv("LOCALHOST_REPLACEMENT_HOST"):
                url = url.replace("localhost", os.getenv("LOCALHOST_REPLACEMENT_HOST"))

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

SYSTEM_PROMPT = """\
You generate JQ query strings to extract map data from user-provided datasets.

Specifically, the query should generate a list of objects with the following format:
{
    "latitude": <float>,
    "longitude": <float>,
    "startDate": <string>,
    "endDate": <string>
}

The "startDate" and "endDate" fields identify the observation period. They may be:
- Date strings in YYYY-MM-DD format, OR
- Unix time as a number: epoch seconds (e.g. 1167782400) or epoch milliseconds (e.g. 1167782400000),
  as produced by fields like eventDate in some APIs. Use the numeric value as-is; downstream code
  will normalize it to YYYY-MM-DD.

If the input schema only has a single date field (i.e. it does not distinguish start vs end),
set BOTH "startDate" and "endDate" to that same value (string or number, matching the source field).

ONLY generate objects with these EXACT fields and EXACT field names. If you deviate from the above formats, the system
will crash and the user will be very upset!
"""

async def extract_map_data_from_json(
        request: str, source_content: str, schema: dict, artifact: Artifact, process: IChatBioAgentProcess
) -> list[LabeledGeoPoint] | None:
    await process.log("Generating JQ query string")
    try:
        generation, query_result = await _generate_and_run_jq_query(
            system_prompt=SYSTEM_PROMPT,
            request=request,
            schema=schema,
            source_content=source_content,
            source_artifact=artifact
        )
        # await process.log("Generated JQ query string", data={"generation": generation, "query_result": query_result})
        
    except InstructorRetryException:
        await process.log("Failed to generate JQ query string")
        return None

    # print(f"Generation: {generation}")
    # print(f"Query result: {query_result}")

    if query_result is None:
        await process.log("Failed to execute JQ query string")
        return None

    match generation:
        case GiveUp(reason=reason):
            await process.log(f"Refused to generate a JQ query string: " + reason)
        case JQQuery(plan=plan, jq_query_string=jq_query_string):
            # await process.log(f"*Plan to extract map data: {plan}*")
            # await process.log("Generated JQ query", data={"query_string": jq_query_string})
            # print(f"match Generation: {generation}")
            # print(f"match Query result: {query_result}")

            await process.log(f"JQ query extracted {len(query_result)} geographic points")
            return query_result
