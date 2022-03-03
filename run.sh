#/!bin/bash

# Allow execution of the main script
chmod +x src/main.py

# Initial question
while true; do
  read -p "------ Hi there, do you want to run the whole pipeline? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
    filename=src/log/"$(date +"%Y_%m_%d_%I_%M_%p").log"
    echo "Wait for a couple of seconds, pipeline is being executed..."
    src/main.py > $filename
    echo "------ Pipeline finished successfully, see below the summary"
    cat $filename
    exit 0
  elif [[ $ans  =~ "n" ]]
  then
    exit 0
  fi  
done


