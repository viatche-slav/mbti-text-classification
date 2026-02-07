from pathlib import Path

import requests


def download_data() -> None:
    """
    Скачивает MBTI датасет из Google Drive.
    """
    url = "https://drive.google.com/uc?\
        export=download&\
        id=1pV09cvKJkPwGLTltIRIoRAFmafxxTbO0"

    project_path = Path(__file__).parent.parent.parent
    output_path = project_path / "data" / "raw" / "mbti_1.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Файл {output_path} уже существует")
        return

    print(f"Скачиваем датасет MBTI в {output_path}")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"Датасет успешно скачан: {output_path}")


if __name__ == "__main__":
    download_data()
