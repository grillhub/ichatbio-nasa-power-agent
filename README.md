# NASA POWER Data Agent

An iChatBio agent that provides access to NASA's Prediction Of Worldwide Energy Resources (POWER) weather and climate data.

## Features

- **Enrich location datasets** — attach NASA POWER data to existing records (e.g. species occurrences, field observations) via artifact input; outputs enriched records with `nasaPowerProperties`
- **Multiple parameters** — temperature (T2M, T2M_MAX, T2M_MIN), humidity (RH2M), wind speed (WS2M), surface pressure (PS), and more; use `list_parameters` for the full catalog

## Available Parameters
Please check here's [Explore POWER's Data Parameters](https://power.larc.nasa.gov/parameters/)

## Testing

### Run Tests

```bash

# Or directly with pytest
python -m pytest -v

# Run specific test file
python -m pytest tests/test_nasa_power_data.py -v
```

### Allure reports

Install the pytest integration:

```bash
python3.13 -m pip install allure-pytest
```

Generate Allure result files for the NASA POWER agent eval suite:

```bash
python3.13 -m pytest -vv tests/evals/eval_assistant/test_nasa_power_agent.py --alluredir=allure-results
```

**View results as a dashboard** — from the project root, start the Allure Docker service (mounts `allure-results` from the current directory):

```bash
cd ~/ichatbio-nasa-power-agent
docker run -d \
  --name allure \
  -p 5050:5050 \
  -e CHECK_RESULTS_EVERY_SECONDS=3 \
  -v $(pwd)/allure-results:/app/allure-results \
  frankescobar/allure-docker-service
```

Open the latest report in your browser: [http://localhost:5050/allure-docker-service/latest-report](http://localhost:5050/allure-docker-service/latest-report)

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
