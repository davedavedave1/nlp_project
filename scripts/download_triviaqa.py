import tarfile
from pathlib import Path
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

URL = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"
ARCHIVE_PATH = DATA_DIR / "triviaqa-rc.tar.gz"

def main():
    print("starting ...")
    # Download if not already there
    if not ARCHIVE_PATH.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, ARCHIVE_PATH)
        print(f"Saved to {ARCHIVE_PATH}")
    else:
        print(f"Archive already exists at {ARCHIVE_PATH}, skipping download.")

    # Extract archive
    print("Extracting archive...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("Done.")

    extracted_dir = DATA_DIR / "triviaqa-rc"
    print(f"Files should now be in: {extracted_dir.resolve()}")

if __name__ == "__main__":
    main()