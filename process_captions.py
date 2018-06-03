import os
import re
import jieba_zhtw as jb

jb.dt.cache_file = 'jieba.cache.zhtw'
cap_path = "train/captions/"
processed_path = "train/p_captions/"

filename_list = os.listdir(cap_path)

for i, filename in enumerate(filename_list) :
    cap_string = ""
    if i % 10 == 0 : print(i)
    try :
        cap_string = open(cap_path + filename, 'r', encoding = 'utf-8-sig').read()
    except :
        cap_string = open(cap_path + filename, 'r').read()
    cap_string = re.sub(": ", "", cap_string)
    cut_cap = jb.cut(cap_string, cut_all = False)
    open(processed_path + filename, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_cap))
