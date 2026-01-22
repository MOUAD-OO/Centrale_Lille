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

    # Ask if user wants to delete a file
    read -p "Do you want to delete a file? [y/N]: " delete_choice
    if [[ "$delete_choice" =~ ^[Yy]$ ]]; then
        read -p "Enter the full path of the file to delete: " file_path
        if [ -f "logs/test.log" ]; then
            rm "logs/test.log"
            echo "File 'logs/test.log' deleted."
        else
            echo "File 'logs/test.log' does not exist."
        fi
    fi

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
    mosquitto_sub \
  -h roger.wizzilab.com \
  -p 8883 \
  -u 2e13dd1e6216 \
  -P e12a34e727286eae0ab251852bef1ae0 \
  -t /applink/2BF12485/location/# \
  -i 2e13dd1e6216:0 \
  -d -v \
  --capath /etc/ssl/certs \
  > logs/test.log

) &
MOSQUITTO_PID=$!

# Run the Python script in the background
python3 Locengine.py --movement "$name" &  
PYTHON_PID=$!

echo "Both processes started in background"
echo "Press Ctrl+C to stop both processes"

# Keep script running
wait
