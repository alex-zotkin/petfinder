import csv
import os.path
import shutil

dict = {
    "eyes": 2,
    "face": 3,
    "near": 4,
    "accessory": 6,
    "group": 7,
    "collage": 8,
    "human": 9,
    "occlusion": 10,
    "info": 11,
    "blur": 12,
}


img_values = {
    "eyes": [0, 0],
    "face": [0, 0],
    "near": [0, 0],
    "accessory": [0, 0],
    "group": [0, 0],
    "collage": [0, 0],
    "human": [0, 0],
    "occlusion": [0, 0],
    "info": [0, 0],
    "blur": [0, 0],
}


with open(r"petfinder-pawpularity-score/train.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)

    count = 0
    for row in reader:
        if count == 0:
            count += 1
            continue

        for k, v in dict.items():
            img_values[k][int(row[v])] += 1

    for k, v in img_values.items():
        img_values[k] = min(v)

    print(img_values)



with open(r"petfinder-pawpularity-score/train.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)

    count = 0
    for row in reader:
        if count == 0:
            count += 1
            continue

        img_name = row[0] + ".jpg"
        for k, v in dict.items():

            to_dir = f"train/{k}/{row[v]}"
            if not os.path.exists(to_dir):
                os.makedirs(to_dir)

            if len(os.listdir(f"{to_dir}")) <= img_values[k]:
                shutil.copyfile(
                    f"petfinder-pawpularity-score/imgs/{img_name}",
                    f"{to_dir}/{img_name}")



with open(r"petfinder-pawpularity-score/test.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)

    count = 0
    for row in reader:
        if count == 0:
            count += 1
            continue

        img_name = row[0] + ".jpg"
        for k, v in dict.items():

            to_dir = f"test/{k}/{row[v]}"
            if not os.path.exists(to_dir):
                os.makedirs(to_dir)

            shutil.copyfile(
                f"petfinder-pawpularity-score/imgs/{img_name}",
                f"{to_dir}/{img_name}")
