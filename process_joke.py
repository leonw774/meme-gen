import os
import re
import jieba_zhtw as jb

jb.dt.cache_file = 'jieba.cache.zhtw'
postpath = "jokes/"
processed_path = "p_jokes/"

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
"→"
]

filename_list = []
for file in os.listdir(postpath) :
    filename_list.append(file)
filename_list = sorted(filename_list)

def tag_remover(post_string) :
    temp = ""
    result = ""
    is_ignored = False
    
    for c in post_string :
        if (c == '\t') :
            continue
        elif (c == ' ' or c == '\n' or c == '\r') :
            if (temp == "") :
                if (result.endswith(' ') or result.endswith('\n') or result.endswith('\r') or result == "") :
                    continue
            elif (temp.endswith(' ') or temp.endswith('\n') or temp.endswith('\r')) :
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
    ingored_result = ""
    for line in result_lines :
        if all(x not in line for x in ignore_list) :
            ingored_result += line
    result = ingored_result

    # ignoring signature
    signature = result.find("--")
    if signature > 0 : result = result[:signature]
    if not result.endswith('\n') : result += '\n'
    return result
# end def tag_remover

for i, filename in enumerate(filename_list) :
    post_string = ""
    if i % 10 == 0 : print(i)
    post_string = open(postpath + filename, 'r', encoding = 'utf-8-sig').read()
    post_string = tag_remover(post_string)
    post_string = re.sub(": ", "", post_string)
    cut_post = jb.cut(post_string, cut_all = False)
    open(processed_path + filename, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_post))
