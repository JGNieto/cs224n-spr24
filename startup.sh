#!/bin/bash

# This script is used to start the GCP instance, since very often starting it
# fails due to a lack of resources, so we retry until it starts successfully.

echo "(1) Attempting to start instance"

i=2
INSTANCE_WAIT=1
INSTANCE_NAME=nvidia-gpu-optimized-vmi-3-vm

while ! gcloud compute instances start ${INSTANCE_NAME};
do
    echo "(${i}) Failed to start instance, retrying in ${INSTANCE_WAIT} seconds"
    i=$((i+1))
    sleep ${INSTANCE_WAIT}
done

echo ""
echo "Instance started successfully after ${i} attempts"

osascript -e "display notification \"Instance ${INSTANCE_NAME} started successfully\" with title \"GCP Instance Started\" "
