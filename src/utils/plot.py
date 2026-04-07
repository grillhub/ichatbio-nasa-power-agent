import geojson
from pydantic import BaseModel


class LabeledGeoPoint(BaseModel):
    latitude: float
    longitude: float
    startDate: str | None = None
    endDate: str | None = None
    date: str | None = None
    value: float | int | str | None = None

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
