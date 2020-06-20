# 0: should not download data from google drive
# 1: should download data from google drive
# Default: 0 - should not download data from google drive
should_download_data=$1

if [ -z "$should_download_data" ]; then
    should_download_data=0
fi

# Download dataset
if [ $should_download_data -eq 1 ]; then
    rm -rf data
    mkdir data
    cd data
    pwd
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1z5ZkDafflgfX0vLqRPFpSeGfBtXx4uuK' -O eng.txt
    cd ..
    pwd
fi

# Create python virtual environment
rm -rf venv
mkdir venv
python3 -m venv venv

# Install python packages
pip install --upgrade pip
pip install -r requirements.txt