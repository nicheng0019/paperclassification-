import re

def clean_string(srcdata):
    remove_chars = [",", ".", ":", "\"", "'", "(", ")", "[", "]", "+"]
    replace_blank_chars = ["\r\n", "-", "/"]

    data = srcdata.strip()

    for char in remove_chars:
        data = data.replace(char, "")
    for char in replace_blank_chars:
        data = data.replace(char, " ")

    pattern = "\s(\d+)"
    data = re.sub(pattern, " ", data)

    while True:
        dstdata = data.replace("  ", " ")
        if dstdata == data:
            break
        data = dstdata

    return dstdata

def check_invalid_word(word):
    import re

    if len(word) < 2:
        return True

    pattern = '^[0-9]+$'
    if re.match(pattern, word) is not None:
        return True

    remove_words = ["page"]
    if word in remove_words:
        return True