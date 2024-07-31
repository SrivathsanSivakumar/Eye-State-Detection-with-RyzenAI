# script to pull dataset from google drive and extract

# general imports
import tarfile
import gdown
import os

def main():
    dataset_path = "data/OACE"
    data_download_path = "data/OACE.tar.gz"

    # get dataset from google drive
    if os.path.isdir(dataset_path):
        print("Dataset already exists")
    else:
        if os.path.exists(data_download_path):
            with tarfile.open(data_download_path, "r:gz") as tar:
                tar.extractall(path="data")
        else:
            download_url = "https://drive.google.com/uc?export=download&id=1Qzuf3M7GOi5_JCmvHopTIe_G4IO7-hjP"
            gdown.download(download_url, data_download_path, quiet=False)
            with tarfile.open(data_download_path, "r:gz") as tar:
                    tar.extractall(path="data")

if __name__ == "__main__":
    main()