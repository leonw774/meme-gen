import os
import re
import platform
import jieba_zhtw as jb

jb.dt.cache_file = 'jieba.cache.zhtw'
joke_path = "train/jokes/"
cap_path = "train/captions/"
p_joke_path = "train/p_jokes/"
p_cap_path = "train/p_captions/"

ignore_list = [
"轉錄",
"Re",
"http",
"看板",
"\r\n.\r\n",
"\r\n]\r\n",
"標題",
"作者",
"時間",
"引述",
"※",
"】",
"【",
"█",
"◆",
"／",
"ˇ",
"ˋ",
"ˊ",
"→"
]

if platform.system() == "Linux" :
    space_character = [' ', '\n', '\r']
else :
    space_character = [' ', '\n']

def tag_remover(post_string) :
    temp = ""
    result = ""
    is_ignored = False
    
    for c in post_string :
        if (c == '\t') :
            continue
        elif (c == ' ' or c == '\n' or c == '\r') :
            if (temp == "") :
                if any(result.endswith(x) for x in space_character) :
                    continue
            elif any(temp.endswith(x) for x in space_character) :
                continue
        
        if not is_ignored :
            if (c == '<') :
                result += temp
                is_ignored = True
                temp = ""
            else :
                temp += c
                if ("&lt;" in temp) :
                    temp = re.sub("&lt;", "<", temp)
                if ("&gt;" in temp) :
                    temp = re.sub("&gt;", ">", temp)
                is_ignored = False
        #end if not is_ignored
        else : #is_ignored
            temp += c
            if (c == '>') :
                is_ignored = False
                temp = ""
        #end else (is_ignored)
    # end for c in post_string
    if temp != "" :
        result += temp

    # ignoring lines for some word
    result_lines = result.split("\n")
    ignored_result = ""
    for line in result_lines :
        if all(x not in line for x in ignore_list) :
            ignored_result += (line + "\n")
    result = ignored_result

    # ignoring signature
    signature = result.find("--")
    if signature > 0 : result = result[:signature]
    if not result.endswith('\n') : result += '\n'
    return result
# end def tag_remover

def process_jokes() :
    filename_list = []
    for file in os.listdir(joke_path) :
        filename_list.append(file)
    filename_list = sorted(filename_list)
    for i, filename in enumerate(filename_list) :
        string = ""
        string = open(joke_path + filename, 'r', encoding = 'utf-8-sig').read()
        string = tag_remover(string)
        string = re.sub(": ", "", string)
        cut_joke = jb.cut(string, cut_all = False)
        open(p_joke_path + filename, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_joke))

def process_captions() :
    filename_list = os.listdir(cap_path)
    for i, filename in enumerate(filename_list) :
        string = ""
        try :
            string = open(cap_path + filename, 'r', encoding = 'utf-8-sig').read()
        except :
            string = open(cap_path + filename, 'r').read()
        cut_cap = jb.cut(string, cut_all = False)
        open(p_cap_path + filename, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_cap))
