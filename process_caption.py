import os
import re
import jieba_zhtw as jb

jb.dt.cache_file = 'jieba.cache.zhtw'
jb.load_userdict('meme_dictionary.dic')

cap_path = "train/captions/"
p_cap_path = "train/p_captions/"
cap_file_list = os.listdir(cap_path)

for i, filename in enumerate(cap_file_list) :
    string = ""
    try :
        string = open(cap_path + filename, 'r', encoding = 'utf-8-sig').read()
    except :
        print(filename)
        string = open(cap_path + filename, 'r').read()
    cut_cap = jb.cut(string, cut_all = False)
    open(p_cap_path + filename, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_cap))
