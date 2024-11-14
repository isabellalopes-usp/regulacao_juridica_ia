import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def checkDirectory(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(directory)
    os.makedirs(directory)

def downloadPages(term, directory):
    searchQuery = f'"{term}"'
    print(f"Searching for term: {term}")
    try:
        url = f"https://esaj.tjsp.jus.br/cjsg/resultadoCompleta.do?dadosConsulta.livre={searchQuery}"
        response = requests.get(url)
        response.raise_for_status()
        with open(os.path.join(directory, f"{term}.html"), 'w', encoding='utf-8') as file:
            file.write(response.text)
    except requests.RequestException as e:
        print("Download failed - ERROR - retrying download...")
        print(e)
        downloadPages(term, directory)

def checkPages(directory, term):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    if len(files) > 0:
        for file in files:
            if os.path.getsize(file) == 0:
                print(f"File {file} is empty. Redownloading...")
                checkDirectory(directory)
                downloadPages(term, directory)
                checkPages(directory, term)
                break

def loadTerms(excelFile):
    df = pd.read_excel(excelFile)
    return df['term'].tolist()

def main():
    terms = loadTerms("terms_utilizados.xlsx")
    for term in terms:
        directory = term.replace(" ", "_")
        checkDirectory(directory)
        downloadPages(term, directory)
        checkPages(directory, term)

if __name__ == "__main__":
    main()
