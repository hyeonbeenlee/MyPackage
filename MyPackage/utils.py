import os
import re


def Sec2Time(seconds):
    Hr = int(seconds // 3600)
    seconds -= Hr * 3600
    Min = int(seconds // 60)
    seconds -= Min * 60
    Sec = seconds
    return Hr, Min, int(Sec)


def CreateDir(DirectoryPath):
    if not os.path.exists(DirectoryPath):
        os.makedirs(DirectoryPath)
        print(f"Created new directory: {DirectoryPath}")
    else:
        pass

def NumFromStr(str):
    list_num = re.findall(r'\d+', str)
    return list_num