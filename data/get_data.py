import os
from pathlib import Path


def main() -> None:
    dataset_name = "kapitanov/easyportrait"
    download_path = "data/portraits"
    Path(download_path).mkdir(exist_ok=True)
    os.system(f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip")


if __name__ == "__main__":
    main()
