def clean_string(srcdata):
    data = srcdata.replace("\r\n", " ")
    data = data.strip()
    data = data.replace(",", "")
    data = data.replace(".", "")
    data = data.replace(":", "")
    data = data.replace("(", "")
    data = data.replace(")", "")
    data = data.replace("[", "")
    data = data.replace("]", "")
    data = data.replace("+", "")
    data = data.replace("-", " ")
    data = data.replace("/", " ")

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