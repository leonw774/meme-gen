from urllib import request

base_url = 'http://www.ghostisland.com.tw/picwar/pic/'

ignore_tag_list = [
"小叮噹",
"塊陶太郎",
"吹捧",
"嘲諷",
"吳敦義",
"淫蕩",
"政治",
""
]
ignore_tag_list = [ ("\">" + s + "</span>") for s in ignore_tag_list]

ignore_cap_list = [
"你", "我", "他"
]

caption_length_min_limit = 8

p = 5
for _ in range(1000) :
    p += 1
    fname = str(p)
    #print("examing: ", fname)
    response = request.urlopen(base_url + fname)
    web_content = str(response.read().decode('utf-8'))
    
    ignore_this = False
    for ignore_tag in ignore_tag_list:
        if web_content.find(ignore_tag) > 0 :
            ignore_this = True
            break
    if ignore_this : continue
    
    title_content = web_content.split("\n")[7]
    if title_content.find("og:description") > 0 :
        continue
    caption_beg = title_content.find("t=\"") + 3
    caption_end = title_content.find("\" /")
    caption_string = title_content[caption_beg : caption_end]
    if any(x in caption_string for x in ignore_cap_list) : continue
    if len(caption_string) < caption_length_min_limit : continue
    
    pic_beg = web_content.find("[img]") + 5
    pic_end = web_content.find("[/img]")
    if pic_end < 0 : continue
    img_url = web_content[pic_beg : pic_end]
    if (".gif" in img_url or ".GIF" in img_url) : continue
    #print(img_url)
    img = request.URLopener()
    img.retrieve(img_url, "./images/" + str(p) + (img_url[-4: ]).lower())
    
    f = open("./captions/" + fname + ".txt", 'w+', encoding = "utf-8-sig")
    f.write(caption_string)
    f.close()
    
    print("downloaded", fname)
