import json

with open("input_data.json", "r") as f:
    data = json.load(f)

print(len(data))
print(data[0].keys())
print(data[0]['stroke_data'].keys())
print(data[0]['stroke_data']['0'])
