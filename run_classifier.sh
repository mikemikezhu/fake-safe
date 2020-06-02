# 0: run all the scripts
# Default: 0 - should run all the scripts
script_name=$1
# 0: should not display samples directly
# 1: should display samples directly
# Default: 0 - should not display samples directly
should_display_directly=$2
# 0: should not save samples to file
# 1: should save samples to file
# Default: 0 - should not save samples to file
should_save_to_file=$3

if [ -z "$script_name" ]; then
    script_name=0
fi

if [ -z "$should_display_directly" ]; then
    should_display_directly=0
fi

if [ -z "$should_save_to_file" ]; then
    should_save_to_file=0
fi

if [ -z "$should_download_data" ]; then
    should_download_data=0
fi

source venv/bin/activate
echo 'Create python virtual environment'

# Create model folder
rm -rf model
mkdir model
cd model
pwd

# Execute python script
cd ..
pwd

if [ $script_name -eq 0 ]; then
    python3 "src/classifier_face_rgb.py" $should_display_directly $should_save_to_file
    python3 "src/classifier_face.py" $should_display_directly $should_save_to_file
    python3 "src/classifier_mnist.py" $should_display_directly $should_save_to_file
    python3 "src/classifier_fashion.py" $should_display_directly $should_save_to_file
else
    python3 "src/$script_name.py" $should_display_directly $should_save_to_file
fi

# Deactivate python virtual environment
deactivate
echo 'Deactivate python virtual environment'