from pathlib import Path


def load_university_data():
    file_path = Path(__file__).resolve().parents[2] / "data" / "university_data.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text