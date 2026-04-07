import json
import logging
from typing import Union

import instructor
import jq
from ichatbio.types import Artifact
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from .plot import LabeledGeoPoint
from .util import JSON

MAX_CHARACTERS_TO_SHOW_AI = 1024 * 10
MAX_SOURCE_PREVIEW_SIZE = 500

NONE = object()

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


class JQQuery(BaseModel):
    plan: str = Field(
        description="A brief explanation of how you plan to query the data (what fields to use, any filters, transformations, etc.)"
    )
    jq_query_string: str = Field(
        description="A JQ query string to process the json artifact.",
    )

class GiveUp(BaseModel):
    reason: str

def _make_validating_response_model(source_content: JSON, results_box: list[list[LabeledGeoPoint] | None]):
    class ValidatedJQQuery(JQQuery):
        @field_validator("jq_query_string", mode="after")
        @classmethod
        def validate_jq_query_string(cls, query):
            # If the LLM doesn't know how to construct an appropriate query, it shouldn't generate one
            if not query:
                raise ValueError("No JQ query string generated. Give up and explain why.")

            try:
                compiled = jq.compile(query)
            except ValueError as e:
                raise ValueError(f"Failed to compile JQ query string {query}", e)

            try:
                result = compiled.input_value(source_content).all()

                # Don't wrap a list in another list
                if type(result) is list and len(result) == 1:
                    result = result[0]

            except ValueError as e:
                raise ValueError(
                    f"Failed to execute JQ query {query} on provided content", e
                )

            if not contains_non_null_content(result):
                raise ValueError(
                    "Executing the JQ query on the input data returned an empty result. Does the query string match the schema of the input data?"
                )

            try:
                parsed_results = [LabeledGeoPoint(**point) for point in result]
            except TypeError as e:
                raise ValueError(
                    f"Failed to parse JQ query result as a list of labeled geo points: {e}"
                )

            results_box[0] = parsed_results

            return query

    class ResponseModel(BaseModel):
        response: Union[ValidatedJQQuery | GiveUp] = Field(
            description="The action you are going to take. If the request can be fulfilled by running a JQ query on data matching the given schema, then you should generate a JQ query. Otherwise, if the request does not make sense with the provided data (e.g. if there are no relevant fields), you should give up and explain why."
        )

    return ResponseModel


async def _generate_and_run_jq_query(
    system_prompt: str, request: str, schema: dict, source_content: JSON, source_artifact: Artifact
) -> tuple[JQQuery | GiveUp, list[LabeledGeoPoint] | None]:
    source_meta = source_artifact.model_dump_json()
    preview = json.dumps(source_content)[:MAX_SOURCE_PREVIEW_SIZE]

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f'The data have the following metadata: "{source_meta}"',
        },
        {
            "role": "user",
            "content": "The JSON data to be processed have the JSON Schema definition that follows. Assume that fields "
            'with no specified "type" are strings.\n\n'
            + json.dumps(schema).replace(': {"type": "string"}', ""),
        },  # Save some tokens
        {
            "role": "user",
            "content": f"For reference, here's the first {len(preview)} characters of the source data: {preview}",
        },
        # {"role": "user", "content": request},
    ]

    results_box: list[list[LabeledGeoPoint] | None] = [None]
    response_model = _make_validating_response_model(source_content, results_box)

    try:
        client: AsyncInstructor = instructor.from_openai(AsyncOpenAI())
        result = await client.chat.completions.create(
            # model="gpt-4.1-unfiltered",
            model="gpt-4.1-mini",
            temperature=0,
            response_model=response_model,
            messages=messages,
            max_retries=2,
        )
    except InstructorRetryException as e:
        logging.warning("Failed to generate JQ query string: %s", e)
        raise

    response: JQQuery | GiveUp = result.response

    return response, results_box[0]
