from typing import Optional, Self, Iterator

import geojson

# geojson visualization tool: https://geojson.io/
def render_points_as_geojson(
    coordinates: list[(float, float)], 
    values: list[float | int | str] = None,
    parameter_data: list[list[dict]] = None
) -> geojson.FeatureCollection:
    features = []
    
    for i, (lat, lon) in enumerate(coordinates):
        if lat is not None and lon is not None:
            properties = {}
            
            if parameter_data is not None and i < len(parameter_data):
                param_list = parameter_data[i]
                if isinstance(param_list, list):
                    for param_info in param_list:
                        if isinstance(param_info, dict):
                            param_name = param_info.get('parameter')
                            if param_name:
                                value = None
                                if 'data' in param_info and isinstance(param_info['data'], list) and len(param_info['data']) > 0:
                                    value = param_info['data'][0].get('value')
                                
                                properties[param_name] = {
                                    "parameter": param_name,
                                    "parameter_description": param_info.get('parameter_description', ''),
                                    "value": value
                                }
            
            if not properties:
                if values is not None and i < len(values):
                    value = values[i]
                else:
                    value = 1.0
                properties = {"value": value}
            
            features.append(
                geojson.Feature(
                    id=i,
                    geometry=geojson.Point((lon, lat)),
                    properties=properties
                )
            )
    
    return geojson.FeatureCollection(features)
