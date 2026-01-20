import requests
from pathlib import Path

def download_gh_file(raw_github_url):
    """
    Download text data from GitHub if it doesn't exist locally.
    
    :param raw_github_url: Path to the file in the GitHub repository
    e.g., https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/refs/heads/master/data/state.csv
    :return: Path to the downloaded file
    """
    data_dir = Path('./data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(raw_github_url).name
    file_path = data_dir / filename
    
    if not file_path.exists():
        print(f"Downloading {filename}...")
        response = requests.get(raw_github_url)
        if response.status_code == 200:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"{filename} downloaded successfully.")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
            return None
    else:
        print(f"{filename} already exists.")
    
    return str(file_path)
