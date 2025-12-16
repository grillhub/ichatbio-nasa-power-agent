# NASA POWER Data Agent

An iChatBio agent that provides access to NASA's Prediction Of Worldwide Energy Resources (POWER) weather and climate data.

## Features

- Query weather and climate data for any location worldwide
- Access multiple parameters including temperature, humidity, precipitation, wind speed, and solar radiation
- Data sourced from NASA's MERRA-2
- Support for multiple locations and date ranges in single queries

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
| `WS2M` | Wind Speed at 2 Meters | m/s |
| `PS` | Surface Pressure | kPa |

## Usage via iChatBio Agent

The agent provides three main entrypoints:

#### 1. List Parameters (`list_parameters`)

Get a complete list of all available NASA POWER parameters with their full descriptions. Use this to discover available weather and climate parameters.

#### 2. Query Weather Data (`query_weather`)

Query weather data for one or more locations and time periods. Supports single location queries, multiple locations, and date ranges. Automatically uses batch processing for optimal performance when querying multiple locations or date ranges.

#### 3. Enrich Locations (`enrich_locations`)

Enrich an existing dataset of location records with NASA POWER weather data. This entrypoint is designed for batch processing of location datasets (e.g., species occurrence records, field observations) that need weather data appended.


## Testing

### Run Tests

```bash

# Or directly with pytest
python -m pytest -v

# Run specific test file
python -m pytest tests/test_nasa_power_data.py -v
```

## Security & Privacy

- No authentication required (public NASA data)
- Read-only access to S3 bucket
- No user data stored
- No API keys required
- All data is publicly available


## References

- [NASA POWER Website](https://power.larc.nasa.gov/)
- [NASA POWER Documentation](https://power.larc.nasa.gov/docs/)
- [iChatBio SDK Documentation](https://github.com/acislab/ichatbio-sdk/)
- [Zarr Documentation](https://zarr.readthedocs.io/)

## License

This project uses publicly available NASA POWER data. Please refer to NASA's data usage policies for more information.

---
