import os
import shutil
from traceback import extract_tb

import numpy as np
import pandas as pd
from PIL import Image

# change what legos should be part of data set
legoShoppingList = [3001,3003,3023,3794,4150,4286,6632,18654,43093,54200]
# chose where the big data set is located on your computer
folder_path = "C:/Users/chris/Downloads/archive/dataset"
# chose where the isolated images will go
new_images = "C:/Users/chris/WPI/Machine Learning/FinalProject/Chris/Chosen2"
# choose the size of the lego images
SIZE = 128



def extractType(lego_name):
    split_name = lego_name.split()
    return int(split_name[0])


filenames = os.listdir(folder_path)
for f in filenames:
    for i in range(0, len(legoShoppingList)):
        if(str(legoShoppingList[i]) in f):
            print("adding image: ", f)
            shutil.copy(os.path.join(folder_path, f), os.path.join(new_images, f))
print("ALL FILES ADDED")

filtered_filenames = os.listdir(new_images)
legoIDs = [extractType(f) for f in filtered_filenames]
df = pd.DataFrame({"filename": filtered_filenames, "ID": legoIDs})
df = df.sort_values(by="ID")
yset = pd.get_dummies(df["ID"]).to_numpy().astype(int)
np.save("Yset.npy", yset)
print("VALIDATION DATA SET CREATED")


xset = []
for f in df["filename"]:
    img_path = new_images + "/" + f
    img = Image.open(img_path).resize((SIZE, SIZE))
    img_array = np.array(img)
    xset.append(img_array)

xset = np.array(xset)
xset = xset[:,:,:,0]

print("INPUT SET CREATED")
np.save("Xset.npy", xset)








