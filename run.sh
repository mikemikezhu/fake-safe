# Create python virtual environment
pwd
rm -rf venv
mkdir venv
python3 -m venv venv
source venv/bin/activate

echo 'Create python virtual environment'

# Install python packages
pip install --upgrade pip
pip install -r requirements.txt

# Execute python script
cd ..
pwd
python3 fake_safe.py

# Deactivate python virtual environment
deactivate
echo 'Deactivate python virtual environment'