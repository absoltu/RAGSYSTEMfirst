from pathlib import Path


def load_text_file(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_path(path: str):

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(path)

    return path_obj