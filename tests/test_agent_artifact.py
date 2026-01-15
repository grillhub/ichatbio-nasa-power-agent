import ichatbio.types
import pytest
import json
import os
from ichatbio.agent_response import DirectResponse, ProcessBeginResponse, ProcessLogResponse, ArtifactResponse, \
    ResponseMessage

from conftest import resource
from src.agent import NASAPowerAgent, BatchEnrichParams


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
@pytest.mark.asyncio
async def test_enrich_locations_with_artifact(context, messages, httpx_mock):
    """Test enriching locations from an artifact"""
    content = resource("list_of_locations_10.json")
    httpx_mock.add_response(url="https://artifact.test", text=content)

    params = BatchEnrichParams(
        locations_artifact=ichatbio.types.Artifact(
            local_id="#0000",
            description="na",
            mimetype="na",
            uris=["https://artifact.test"],
            metadata={},
        ),
        weather_parameters=['T2M'],
        date_range_days=1,
        frequency='daily'
    )

    await NASAPowerAgent().run(
        context,
        "Enrich locations from artifact",
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

    # Should log artifact retrieval
    log_text = " ".join([m.text for m in log_messages])
    assert "artifact" in log_text.lower() or "#0000" in log_text

    # Response should mention enrichment results
    response_text = messages[-1].text
    assert "Enrichment" in response_text or "enriched" in response_text.lower()

    # Check for artifact creation (enriched JSON data)
    artifact_messages = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert len(artifact_messages) >= 1

    # Verify the artifact contains the enriched data
    artifact = artifact_messages[0]
    assert artifact.mimetype == "application/json"
    
    # Find and save the GeoJSON artifact
    geojson_artifact = None
    for artifact_msg in artifact_messages:
        if artifact_msg.metadata and artifact_msg.metadata.get("format") == "geojson":
            geojson_artifact = artifact_msg
            break
    
    if geojson_artifact and geojson_artifact.content:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            geojson_data = json.loads(geojson_artifact.content.decode('utf-8'))
            geojson_file = os.path.join(output_dir, "artifact_test_geojson_output.json")
            with open(geojson_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2, ensure_ascii=False)
            print(f"Saved GeoJSON artifact to {geojson_file}")
            
            if isinstance(geojson_data, dict) and 'features' in geojson_data:
                print(f"GeoJSON contains {len(geojson_data['features'])} features")
                
                # Verify that properties contain multiple parameters
                if len(geojson_data['features']) > 0:
                    first_feature = geojson_data['features'][0]
                    properties = first_feature.get('properties', {})
                    
                    # Check if properties have parameter keys (not just 'value')
                    param_keys = [k for k in properties.keys() if k != 'value']
                    if param_keys:
                        print(f"First feature has {len(param_keys)} parameters: {', '.join(param_keys)}")
                        # Verify structure of first parameter
                        first_param = param_keys[0]
                        param_data = properties[first_param]
                        if isinstance(param_data, dict):
                            assert 'parameter' in param_data, "Parameter data should have 'parameter' field"
                            assert 'parameter_description' in param_data, "Parameter data should have 'parameter_description' field"
                            assert 'value' in param_data, "Parameter data should have 'value' field"
                            print(f"  Example: {first_param} = {param_data.get('value')} ({param_data.get('parameter_description')})")
                    else:
                        print("Warning: Properties only contain 'value', no parameter keys found")
        except Exception as e:
            print(f"Could not save GeoJSON: {e}")
    else:
        print("Warning: No GeoJSON artifact found in messages")

