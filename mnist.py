import requests
from pathlib import Path


def main():
    # Create directory 'data/mnist/'
    (data_path := Path("data", "mnist")).mkdir(parents=True, exist_ok=True)

    # Create the path for the data
    file_path = data_path / "mnist.pkl.gz"
    
    # Raise if it already exists
    if file_path.exists():
        raise FileExistsError
    
    # Open the file, download its contents
    with file_path.open("wb") as file:
        data = requests.get("https://github.com/pytorch/tutorials/raw/main/_static/mnist.pkl.gz").content
        file.write(data)

if __name__ == "__main__":
    main()