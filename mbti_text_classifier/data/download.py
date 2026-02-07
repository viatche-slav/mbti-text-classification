from pathlib import Path

import requests


def download_data() -> None:
    """
    Скачивает MBTI датасет из открытого источника (Kaggle via Google Drive).
    Этот файл затем будет добавлен в DVC для версионирования.
    """
    # Прямая ссылка на датасет MBTI с Google Drive
    # (это публичная ссылка на датасет с Kaggle)
    file_id = "1pV09cvKJkPwGLTltIRIoRAFmafxxTbO0"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Определяем путь относительно корня проекта
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "raw" / "mbti_1.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Файл {output_path} уже существует, пропускаем загрузку")
        return

    print(f"Скачиваем датасет MBTI в {output_path}...")

    # Скачиваем файл
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    # Сохраняем
    output_path.write_bytes(response.content)
    print(f"Датасет успешно скачан: {output_path}")


if __name__ == "__main__":
    download_data()
