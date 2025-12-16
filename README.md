# NASA POWER Data Agent

An iChatBio agent that provides access to NASA's Prediction Of Worldwide Energy Resources (POWER) weather and climate data.

## Features

- 🌍 Query weather and climate data for any location worldwide
- 📊 Access multiple parameters including temperature, humidity, precipitation, wind speed, and solar radiation
- ⏰ Retrieve data at hourly, daily, or monthly frequencies
- 🛰️ Data sourced from NASA's MERRA-2 reanalysis dataset (1980-present)
- 📈 Automatic statistical analysis of retrieved data
- ⚡ Performance-optimized with dataset caching
- 🔒 Public data, no authentication required

## Quick Start

*Requires Python 3.12 or higher*

### Installation

```bash
# Clone or navigate to project directory
cd /path/to/ichatbio-nasa-power-agent

# Install dependencies (this also installs the package)
pip install -e .
```

### Run the Server

```bash
# Start the agent server (CORRECT command)
uvicorn src.agent:create_app --factory --reload --host "0.0.0.0" --port 9999
```

**Note**: The command in older documentation using `--app-dir src` doesn't work due to relative import issues. Use the command above.

### Verify It's Working

Visit: http://localhost:9999/.well-known/agent.json

You should see the agent card with available entrypoints.

### Run with Docker

```bash
docker compose up --build
```

## Available Parameters

| Code | Description | Unit |
|------|-------------|------|
| `T2M` | Temperature at 2 Meters | °C |
| `T2M_MAX` | Maximum Temperature at 2 Meters | °C |
| `T2M_MIN` | Minimum Temperature at 2 Meters | °C |
| `RH2M` | Relative Humidity at 2 Meters | % |
| `PRECTOTCORR` | Precipitation Corrected | mm/day |
| `WS2M` | Wind Speed at 2 Meters | m/s |
| `PS` | Surface Pressure | kPa |
| `ALLSKY_SFC_SW_DWN` | All Sky Surface Shortwave Downward Irradiance | kW-hr/m²/day |

## Usage

### Option 1: Via iChatBio Agent

The agent provides two main entrypoints:

#### 1. Query Weather Data

Query weather data for a specific location and time period.

**Example Request:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "parameter": "T2M",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "frequency": "daily"
}
```

**Parameters:**
- `latitude` (required): Latitude coordinate (-90 to 90)
- `longitude` (required): Longitude coordinate (-180 to 180)
- `parameter` (required): Weather parameter (e.g., "T2M", "RH2M")
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `frequency` (optional): "hourly", "daily", or "monthly" (default: "daily")

**Response:**
- Summary with location, date range, and statistics
- JSON artifact containing full dataset with all data points

#### 2. List Parameters

Get a list of all available weather parameters.

**Parameters:** None

### Option 2: Direct Python Usage

You can use the data fetching module directly in your Python code:

```python
from src.nasa_power_data import NASAPowerDataFetcher

# Initialize fetcher
fetcher = NASAPowerDataFetcher()

# Fetch data
data = fetcher.get_data(
    start_date="2024-01-01",
    end_date="2024-01-31",
    latitude=40.7128,
    longitude=-74.0060,
    parameter="T2M",
    frequency="daily"
)

# Access results
print(f"Mean temperature: {data['statistics']['mean']:.2f}°C")
print(f"Min: {data['statistics']['min']:.2f}°C")
print(f"Max: {data['statistics']['max']:.2f}°C")

# Access individual data points
for point in data['data']:
    print(f"{point['date']}: {point['value']:.2f}°C")
```

### Response Structure

```python
{
    "parameter": "T2M",
    "parameter_description": "Temperature at 2 Meters (°C)",
    "frequency": "daily",
    "latitude": 40.71875,          # Snapped to nearest grid point
    "longitude": -74.0625,          # Snapped to nearest grid point
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "count": 31,
    "data": [
        {"date": "2024-01-01", "value": 5.23},
        {"date": "2024-01-02", "value": 6.45},
        ...
    ],
    "statistics": {
        "mean": 5.67,
        "min": -2.34,
        "max": 12.45,
        "std": 3.21
    }
}
```

## Examples

### Example 1: Temperature Trend for a City

```python
from src.nasa_power_data import NASAPowerDataFetcher

fetcher = NASAPowerDataFetcher()

# Get daily temperatures for New York City in January 2024
data = fetcher.get_data(
    start_date="2024-01-01",
    end_date="2024-01-31",
    latitude=40.7128,
    longitude=-74.0060,
    parameter="T2M",
    frequency="daily"
)
```

### Example 2: Compare Multiple Locations

```python
# Get precipitation for two cities
nyc_data = fetcher.get_data(
    start_date="2024-01-01",
    end_date="2024-01-31",
    latitude=40.7128,
    longitude=-74.0060,
    parameter="PRECTOTCORR",
    frequency="daily"
)

london_data = fetcher.get_data(
    start_date="2024-01-01",
    end_date="2024-01-31",
    latitude=51.5074,
    longitude=-0.1278,
    parameter="PRECTOTCORR",
    frequency="daily"
)
```

### Example 3: Multi-Year Climate Analysis

```python
# Get monthly average temperatures for climate analysis
data = fetcher.get_data(
    start_date="2020-01-01",
    end_date="2024-12-31",
    latitude=35.6762,
    longitude=139.6503,
    parameter="T2M",
    frequency="monthly"
)
```

### Run Example Script

```bash
python example_usage.py
```

## Testing

### Run Tests

```bash
# Using the test script
bash run_tests.sh

# Or directly with pytest
python -m pytest -v

# Run specific test file
python -m pytest tests/test_nasa_power_data.py -v
```

**Note**: Some tests are marked as skipped because they require actual S3 access and network calls. This is intentional to keep unit tests fast. The skipped tests can be run manually when testing with real data.

## Common Locations (Coordinates)

| City | Latitude | Longitude |
|------|----------|-----------|
| New York | 40.7128 | -74.0060 |
| London | 51.5074 | -0.1278 |
| Tokyo | 35.6762 | 139.6503 |
| Paris | 48.8566 | 2.3522 |
| Sydney | -33.8688 | 151.2093 |
| São Paulo | -23.5505 | -46.6333 |
| Mumbai | 19.0760 | 72.8777 |
| Cairo | 30.0444 | 31.2357 |

## Architecture Overview

```
┌─────────────────────────────────┐
│   iChatBio Platform             │
└────────────┬────────────────────┘
             │ HTTP/JSON
┌────────────▼────────────────────┐
│   NASA POWER Agent (agent.py)  │
│   - query_weather               │
│   - list_parameters             │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Data Fetcher                  │
│   (nasa_power_data.py)          │
│   - Validate inputs             │
│   - Connect to S3/Zarr          │
│   - Extract & process data      │
└────────────┬────────────────────┘
             │ s3fs + zarr
┌────────────▼────────────────────┐
│   AWS S3 (Public Bucket)        │
│   NASA POWER MERRA-2 Dataset    │
│   - Global coverage             │
│   - 1980-present                │
│   - Zarr format                 │
└─────────────────────────────────┘
```

### How It Works

1. User makes request via iChatBio with location and parameters
2. Agent validates all input parameters (dates, coordinates, frequency)
3. Data fetcher connects to appropriate Zarr dataset on S3
4. Coordinates are mapped to nearest grid point in dataset
5. Time range is converted to days since reference date (1980-01-01)
6. Data is extracted for the specified location and time range
7. Statistics are calculated (mean, min, max, std dev)
8. Results are formatted as JSON artifact and summary
9. Agent returns response to user via iChatBio

## Data Source

**NASA POWER (Prediction Of Worldwide Energy Resources)**
- Dataset: MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)
- Storage: AWS S3 public bucket (no authentication required)
- Format: Zarr (cloud-optimized for efficient access)
- Coverage: Global (all latitudes and longitudes)
- Time Range: 1980-01-01 to present (near real-time updates)
- Resolution: ~0.5 degrees (~50km at equator)

## Error Handling

The agent validates inputs and provides clear error messages:

| Error Type | Cause | Solution |
|------------|-------|----------|
| Invalid frequency | Wrong frequency value | Use: `hourly`, `daily`, or `monthly` |
| Invalid date format | Wrong date format | Use format: `YYYY-MM-DD` |
| Invalid date range | start_date > end_date | Ensure start_date < end_date |
| Invalid coordinates | Out of bounds | Lat: -90 to 90, Lon: -180 to 180 |
| Parameter not found | Typo or unavailable parameter | Check parameter list with list_parameters |
| No data found | Date out of range | Use dates from 1980-01-01 onwards |

## Performance Notes

1. **Dataset Caching**: The fetcher caches opened Zarr datasets to improve performance for multiple queries
2. **Efficient Data Transfer**: Only the required time slices and spatial points are fetched, minimizing data transfer
3. **Spatial Resolution**: The dataset has ~0.5 degree resolution; coordinates are snapped to nearest grid point
4. **First Query Latency**: First query to a dataset frequency takes longer due to metadata loading
5. **Large Time Ranges**: Consider using monthly frequency for multi-year queries to reduce query time

## Troubleshooting

### Issue: "No data found for the specified date range"
- Check that your dates are within the dataset range (1980-01-01 to present)
- Verify date format is YYYY-MM-DD
- Ensure start_date is before end_date

### Issue: "Parameter not found in dataset"
- Use the `list_parameters` entrypoint to see available parameters
- Parameter names are case-sensitive (e.g., "T2M" not "t2m")

### Issue: Slow data fetching
- Use coarser frequency (monthly instead of daily/hourly) for long time ranges
- Reduce date range
- The first query to a dataset takes longer due to metadata loading

### Issue: ImportError when running server
- Make sure you've installed the package: `pip install -e .`
- Use the correct command: `uvicorn src.agent:create_app --factory --reload --host "0.0.0.0" --port 9999`
- Do NOT use `--app-dir src` (causes relative import issues)

### Issue: zarr version conflicts
- Ensure zarr 2.18.x is installed (not 3.x)
- Reinstall: `pip uninstall zarr -y && pip install "zarr~=2.18.0"`

## Project Structure

```
ichatbio-nasa-power-agent/
├── src/
│   ├── agent.py              # iChatBio agent implementation
│   ├── nasa_power_data.py    # NASA POWER data fetching module
│   └── __init__.py           # Package initialization
├── tests/
│   ├── test_agent.py         # Agent tests
│   └── test_nasa_power_data.py  # Data fetcher tests
├── example_usage.py          # Example usage script
├── run_tests.sh              # Test runner script
├── pyproject.toml            # Project dependencies
├── Dockerfile                # Docker configuration
├── compose.yaml              # Docker compose configuration
└── README.md                 # This file
```

## Dependencies

Key dependencies (automatically installed with `pip install .`):
- `ichatbio-sdk~=0.2.1` - iChatBio agent framework
- `zarr~=2.18.0` - Zarr array format
- `s3fs~=2024.6.1` - S3 filesystem interface
- `numpy~=1.26.0` - Numerical operations
- `pydantic~=2.11.4` - Data validation
- `uvicorn~=0.34.3` - ASGI server
- `pytest~=8.3.5` - Testing framework

## Useful Commands

```bash
# Install dependencies
pip install -e .

# Run example
python example_usage.py

# Run tests
bash run_tests.sh
# or
python -m pytest -v

# Start server
uvicorn src.agent:create_app --factory --reload --host "0.0.0.0" --port 9999

# Docker
docker compose up --build
```

## Security & Privacy

- ✅ No authentication required (public NASA data)
- ✅ Read-only access to S3 bucket
- ✅ No user data stored
- ✅ No API keys required
- ✅ All data is publicly available

## Limitations

1. **Spatial Resolution**: ~0.5 degree grid (~50km), coordinates snapped to nearest point
2. **Date Range**: Limited to 1980-01-01 onwards
3. **Data Latency**: Near real-time but may lag by a few days
4. **Network Dependency**: Requires internet access to AWS S3

## Future Enhancements

Potential improvements:
- Add more parameters from full NASA POWER dataset
- Implement data caching for frequently requested locations
- Add visualization artifacts (charts, maps)
- Support for bounding box queries (area instead of point)
- Historical trend analysis and anomaly detection
- Export to different formats (CSV, NetCDF)

## References

- [NASA POWER Website](https://power.larc.nasa.gov/)
- [NASA POWER Documentation](https://power.larc.nasa.gov/docs/)
- [MERRA-2 Dataset](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)
- [iChatBio SDK Documentation](https://github.com/ichatbio/sdk)
- [Zarr Documentation](https://zarr.readthedocs.io/)

## License

This project uses publicly available NASA POWER data. Please refer to NASA's data usage policies for more information.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the example_usage.py script for working examples
3. Ensure all dependencies are correctly installed
4. Verify you're using Python 3.12 or higher

---

**Status**: ✅ Fully functional and ready to use!
