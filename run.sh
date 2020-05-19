# Shell script arguments
# Python script name
script_name=$1
# 0: should not reset
# 1: should reset, create virtual environment and install Python packages
# Default: 0 - should not reset
should_reset=$2
# 0: should not display samples directly
# 1: should display samples directly
# Default: 0 - should not display samples directly
should_display_directly=$3
# 0: should not save samples to file
# 1: should save samples to file
# Default: 0 - should not save samples to file
should_save_to_file=$4

if [ -z "$script_name" ]; then
    echo "Please provide python script name in the first argument"
    exit 1
else
    echo "Run script: $script_name"
fi

if [ -z "$should_reset" ]; then
    should_reset=0
fi

if [ -z "$should_display_directly" ]; then
    should_display_directly=0
fi

if [ -z "$should_save_to_file" ]; then
    should_save_to_file=0
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
python3 "$script_name.py" $should_display_directly $should_save_to_file

# Deactivate python virtual environment
deactivate
echo 'Deactivate python virtual environment'