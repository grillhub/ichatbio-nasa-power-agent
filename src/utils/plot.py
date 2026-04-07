import geojson
from pydantic import BaseModel


class LabeledGeoPoint(BaseModel):
    latitude: float
    longitude: float
    startDate: str | int | float | None = None
    endDate: str | int | float | None = None
    date: str | int | float | None = None

def render_points_as_geojson(
    map_data: list[LabeledGeoPoint]
) -> geojson.FeatureCollection:
    geo = geojson.FeatureCollection(
        [
            geojson.Feature(
                id=i, geometry=geojson.Point((point.longitude, point.latitude)), properties={"value": point.value}
            )
            for i, point in enumerate(map_data)
            if point.latitude is not None and point.longitude is not None
        ]
    )

    return geo
