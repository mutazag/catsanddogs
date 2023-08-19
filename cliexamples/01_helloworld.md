az ml job create -f 01_helloworld.yml


az ml job create -f 01_helloworld.yml --set environment.image="library/python:3.8"

az ml job create -f 02_helloworld_inputs.yml --set environment.image="library/python:3.8"