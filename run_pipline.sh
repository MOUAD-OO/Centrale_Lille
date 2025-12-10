#!/bin/bash

echo "This script will get position values and visualize them, used for collecting movement vectors."
echo "======================================================================================"
echo "             Make sure the anchor JSON is correct and updated                         "
echo "======================================================================================"
read -p "Enter the movement type: " name

if [ -z "$name" ]; then
  echo "Error: movement type cannot be empty."
  exit 1
fi

echo "Collecting '$name' test vectors"
echo ">>> Starting MQTT listener (press CTRL+C to stop)..."



# Run mosquitto_sub until manual stop

mosquitto_sub \
  -h roger.wizzilab.com \
  -u 634b8f91f3ff \
  -P 9422773ebafbef2d074cd2be912d063d \
  -p 8883 \
  -t /applink/27C6B975/location/1506/# \
  -i 634b8f91f3ff:0 \
  -d -v \
  --capath /etc/ssl/certs \
  > /logs/test.log

echo ">>> MQTT stopped."
echo "Log saved to: logs/test.log"

echo ">>> Running distance preprocessing..."

python /home/nathan/Dynamic_Loc/data_preparing/get_distance.py -movement "$name"
if [ $? -ne 0 ]; then
  echo "Error: distance preprocessing failed."
  exit 1
fi

echo ">>> Running prediction..."

python -m data_preparing.predict_path \
  --input "test_site/${name}_001BC50C70027E04.csv" \
  --output "test_site/${name}_001BC50C70027E04.npy"
if [ $? -ne 0 ]; then
  echo "Error: prediction failed."
  exit 1
fi

echo ">>> Running analysis..."

python3 analyse.py \
  --real "test_site/${name}.npy" \
  --est1 "test_site/${name}_v7.npy"\
  --est2 "test_site/${name}_v7.npy"
if [ $? -ne 0 ]; then
  echo "Error: analysis failed."
  exit 1
fi

echo ">>> Pipeline finished successfully!"


