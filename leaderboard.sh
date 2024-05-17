#!/bin/bash

# Check if exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <test|dev>"
    exit 1
fi

# Set variables based on the argument
if [ "$1" == "dev" ]; then
    rm -f dev_files.zip
    output_file="dev_files.zip"
    files_to_zip=("predictions/sst-dev-output.csv" "predictions/sts-dev-output.csv" "predictions/para-dev-output.csv")
elif [ "$1" == "test" ]; then
    # Ask for confirmation
    echo "WE CAN ONLY SUBMIT TO TEST 3 TIMES!!! USE DEV TO TEST YOUR MODEL!!!"
    echo "Type 'understood' to confirm that you are submitting to the test set."
    read confirmation
    if [ "$confirmation" != "understood" ]; then
        echo "Confirmation not received. Exiting."
        exit 1
    fi

    rm -f test_files.zip
    output_file="test_files.zip"
    files_to_zip=("predictions/sst-test-output.csv" "predictions/sts-test-output.csv" "predictions/para-test-output.csv")
else
    echo "Invalid argument. Please use 'test' or 'dev'."
    exit 1
fi

# Create the zip file
zip $output_file "${files_to_zip[@]}"

# Confirm creation of the zip file
if [ $? -eq 0 ]; then
    echo "$output_file has been created successfully."
else
    echo "Failed to create $output_file."
    exit 1
fi
