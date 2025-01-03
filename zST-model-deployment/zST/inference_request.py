import requests
import time
import argparse
import json

CLI = argparse.ArgumentParser()
CLI.add_argument("--model", type=str, default="rf_model")
args = CLI.parse_args()

file1 = open('input_batch_pos.json', 'r')
input_batch_pos = json.loads(file1.readlines()[0])
#print(input_batch_pos)
jason_data = input_batch_pos
inference_endpoint = "http://localhost:8000/v2/models/rf_model/infer"


response = requests.post(inference_endpoint, json=jason_data)
print(response.text)
