from pathlib import Path

import requests


def download_data(data_file, url):
    data_path = Path(data_file)

    if data_path.exists():
        return

    data_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data_path.write_bytes(response.content)
