#!/bin/bash

echo "This script will get position values and visualize them, used for collecting movement vectors."
echo "======================================================================================"
echo "             Make sure the anchor JSON is correct and updated                         "
echo "======================================================================================"

# Trap to handle stopping both processes
cleanup() {
    echo ""
    echo "Stopping both processes..."
    kill -TERM $MOSQUITTO_PID $PYTHON_PID 2>/dev/null
    sleep 1
    kill -KILL $MOSQUITTO_PID $PYTHON_PID 2>/dev/null
    wait 2>/dev/null
    echo "Processes stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

read -p "Enter the movement type: " name

if [ -z "$name" ]; then
  echo "Error: movement type cannot be empty."
  exit 1
fi

echo "Collecting '$name' test vectors"

# Start mosquitto_sub in the background
(
    mosquitto_sub  \
    -h roger.wizzilab.com \
    -u 634b8f91f3ff \
    -P 9422773ebafbef2d074cd2be912d063d \
    -p 8883 \
    -t /applink/27C6B975/location/1506/# \
    -i 634b8f91f3ff:0 \
    -d -v \
    --capath /etc/ssl/certs \
    > logs/test.log
) &
MOSQUITTO_PID=$!

# Run the Python script in the background
python3 mr_robot.py --movement "$name" &  
PYTHON_PID=$!


echo "Both processes started in background"
echo "Press Ctrl+C to stop both processes"

# Keep script running
wait