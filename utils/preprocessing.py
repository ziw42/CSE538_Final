### Format the downloaded FakeNewsNet data to our own data files
import json
import os

base_path = "/home/zianwang/fknews/utils/FakeNewsNet/code/fakenewsnet_dataset/"
data = {}
for dataset in ["gossipcop", "politifact"]:
    folder_path = base_path + dataset
    data[dataset] = {"fake": [],
                     "real": []}
    for split in ["fake", "real"]:
        sub_path = folder_path + "/" + split
        for file_dir in os.listdir(sub_path):
            file_dir = sub_path + "/" + file_dir
            for file in os.listdir(file_dir):
                file_path = file_dir + "/" + file
                with open(file_path, "r", encoding="utf-8") as f:
                    line = json.load(f)
                    data[dataset][split].append(line["text"])

save_path = "/home/zianwang/fknews/data/FakeNewsNet_Data.jsonl"
with open(save_path, 'w', encoding='utf-8') as file:
    for dict_item in data:
        for split in data[dict_item]:
            for line in data[dict_item][split]:
                if len(line) != 0:
                    obj = {"text": line, "label": split, "dataset": dict_item}
                    json_line = json.dumps(obj) + '\n'
                    file.write(json_line)
print("Finish")
    






