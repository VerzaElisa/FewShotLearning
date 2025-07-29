from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd
import os
import time
from tqdm import tqdm
import requests
from pathlib import Path
from bs4 import BeautifulSoup

#Funzione per generare i link per il download delle registrazioni complete
def link_generator(row, base_url):
    link_list = []
    url = base_url+str(row['filename'])+".zip"
    link_list.append(url)
    return link_list

# Funzione per ritornare una lista di elementi presenti in un tag select html
def list_pages(web_elem, id, driver):
    elem_list = []
    for i in range(len(web_elem)):
        elem_select = Select(driver.find_element("name", id))
        elem_options = elem_select.options[1:]
        option = elem_options[i]
        elem_name = option.text.strip()
        elem_list.append(elem_name)
    return elem_list

# Funzione per scaricare i file del dataset
def download_dataset(ds_path, url, spec_id, year_id):
    driver = webdriver.Chrome()
    driver.get(url)
    species_select = Select(driver.find_element("name", spec_id))
    species_options = species_select.options[1:]
    species_name = list_pages(species_options, spec_id, driver)
    print(f"Trovate {len(species_name)} specie.")
    
    for spec in species_name:
        # Selezione della specie
        print(f"Elaborazione specie: {spec}")
        species_select = Select(driver.find_element("name", spec_id))
        species_select.select_by_visible_text(spec)
        time.sleep(1) 
        os.makedirs(spec, exist_ok=True)

        # Creazione lista per gli anni relativi alla specie
        year_select = Select(driver.find_element("name", year_id))
        year_options = year_select.options[1:]
        year_list = list_pages(year_options, year_id, driver)
        print(f"Elaborazione anni per {spec}: {year_list}")

        for year in year_list:
            # Selezione dell'anno
            year_select = Select(driver.find_element("name", year_id))
            year_select.select_by_visible_text(year)
            time.sleep(2) 

            # Creazione della lista con i link per il download dei file
            download_links = driver.find_elements("partial link text", "Download")
            print(f"Anno {year}: trovati {len(download_links)} file.")

            # Download dei file
            for link in tqdm(download_links, desc="Download"):
                file_url = link.get_attribute("href")
                file_name = file_url.split('/')[-1]
                file_path = Path(ds_path, spec, file_name)

                # Scarica il file
                if os.path.exists(file_path):
                    continue
                response = requests.get(file_url)
                with open(file_path, 'wb') as f:
                    f.write(response.content)

# Funzione per l'estrazione dei metadati
def retrieve_metadata(folder_path, base_url, col):
    ret_list = []
    for f in os.listdir(folder_path):
        fn = f.split('.')[0]
        curr_dict = {}
        curr_url = base_url + fn
        curr_dict['filename'] = fn
        curr_dict['species'] = f

        # Richiesta html e parsing della tabella
        html = requests.get(curr_url)
        soup = BeautifulSoup(html.content, 'html.parser')
        rows = soup.find_all('tr')

        # Estrazione dei dati dalla tabella
        campi_desiderati = col[1:]
        for row in rows:
            celle = row.find_all('td')
            if len(celle) == 2:
                chiave = celle[0].get_text(strip=True)
                valore = celle[1].get_text(strip=True)
                if chiave in campi_desiderati:
                    curr_dict[chiave] = valore
        ret_list.append(curr_dict)

    return ret_list

# Verifica della presenza dei file audio completi per ogni anno di ogni specie
def check_full_audio(row, ret_list, base_url):
    url = base_url + row['filename']+".zip"
    curr_list = []
    request = requests.get(url, stream=True, allow_redirects=True, timeout=10)
    if request.status_code == 200:
        curr_list = [row['species'], row['filename'], True]
        print(f'{row['species']} - {row['filename']}: True')
    else:
        curr_list = [row['species'], row['filename'], False]
        print(f'{row['species']} - {row['filename']}: False')
    time.sleep(2)
    return ret_list.append(curr_list)

