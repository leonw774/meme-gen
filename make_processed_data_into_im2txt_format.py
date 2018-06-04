import os
import re

image_file_list = [(i) for i in os.listdir("train/images/")]
caption_file_list = [("train/p_captions/" + i) for i in os.listdir("train/p_captions/")]

clean_captions_file = open("CaptionClean.txt", "w+", encoding = "utf-8-sig")
ordered_meme_file = open("ordered_memes.txt", "w+", encoding = "utf-8-sig")

for cap_file in caption_file_list :
    try :
        string = open(cap_file, 'r', encoding = 'utf-8-sig').read()
    except :
        string = open(cap_file, 'r').read()
    string = re.sub(r"\n", " ", string)
    clean_captions_file.write(string + "\n")
    
for image_name in image_file_list :
    ordered_meme_file.write(image_name + "\n")