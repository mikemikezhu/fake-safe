# Shell script arguments
# Python script name
script_name=$1
# 0: should not reset
# 1: should reset, create virtual environment and install Python packages
should_reset=$2

if [ -z "$script_name" ]; then
    echo "Please provide python script name in the first argument"
    exit 1
else
    echo "Run script: $script_name"
fi

if [ -z "$should_reset" ]; then
    should_reset=0
fi

# Create python virtual environment
if [ $should_reset -eq 1 ]; then
    rm -rf venv
    mkdir venv
    python3 -m venv venv
fi
source venv/bin/activate
echo 'Create python virtual environment'

# Install python packages
if [ $should_reset -eq 1 ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Create output folder
rm -rf output
mkdir output
cd output
pwd

# Execute python script
cd ..
pwd
python3 "$script_name.py"

# Deactivate python virtual environment
deactivate
echo 'Deactivate python virtual environment'