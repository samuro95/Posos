import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import re

def get_med_list():
    meds=[]
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for letter in letters :
        meds_letter = []
        url = "https://www.vidal.fr/Sommaires/Medicaments-"+letter+".htm"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for line in soup.get_text().split("\n"):
            if len(line)>2 and line[0]==letter and line[1] in letters :
                line = re.split(' |/|,',line)
                if len(line)>3 and line[0] not in meds_letter:
                    meds_letter.append(line[0])
        meds = meds + meds_letter
    return(meds)
