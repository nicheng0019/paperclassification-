def clean_string(srcdata):
    data = srcdata.replace("\r\n", " ")
    data = data.strip()
    data = data.replace(",", "")
    data = data.replace("(", "")
    data = data.replace(")", "")
    data = data.replace("[", "")
    data = data.replace("]", "")
    data = data.replace("+", "")
    data = data.replace("-", " ")

    while True:
        dstdata = data.replace("  ", " ")
        if dstdata == data:
            break
        data = dstdata

    return dstdata