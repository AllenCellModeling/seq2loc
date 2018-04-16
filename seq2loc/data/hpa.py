import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tqdm

from bs4 import BeautifulSoup
import re

import numpy as np

################################################
### A bunch of scripts for scraping from HPA ###
################################################


def get_single_channel_urls(multi_channel_url, channel_map={'blue':'nucleus', 'red':'microtubules', 'green':'antibody'}):
    parname, bname = os.path.split(multi_channel_url)
    name, ext = bname.split('.')
    plate, well, field = name.split("_")[:3]
    channels = name.split("_")[3:]
    single_channel_urls = {}
    for channel in channels:
        sc_name = '_'.join([plate, well, field, channel])
        sc_bname = '.'.join([sc_name,ext])
        sc_url = os.path.join(parname,sc_bname)
        single_channel_urls[channel_map[channel]] = sc_url
    seg_name = '_'.join([plate, well, field, 'segmentation'])
    seg_bname = '.'.join([seg_name,'png'])
    seg_url = os.path.join(parname,seg_bname)
    return {'singleChannelUrls':single_channel_urls, 'segmentationUrl':seg_url}

def make_xml_tree_from_url(ensembl_input_url):
    req = urllib.request.Request(ensembl_input_url)
    xml = urllib.request.urlopen(req).read()
    return ET.fromstring(xml.decode("utf-8"))

def filter_element(element, tag):
    child = [x for x in list(element) if x.tag == tag]
    child = child[0] if len(child)==1 else None
    return child



def ensg_to_antibody_html(ensg_id):
    '''Gets the html corresponding to an antibody that targets the ensg_id'''
    url = 'https://www.proteinatlas.org/{}/antibody'.format(ensg_id)

    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    html_bin = response.read()
    
    return html_bin
#     soup = BeautifulSoup(html_bin, 'html.parser')

def antibody_and_html_to_ensp(antibody_id, antibody_html_bin):
    '''Gets the ensp_ids that are targeted by the antibody_id'''
    
    antibody_html_str = str(antibody_html_bin)
    
    soup = BeautifulSoup(antibody_html_bin, 'html.parser')
    
    souplets = soup.find_all('th', attrs={'class':'head last roundtop'})

    antibody_ids = list()

    for souplet in souplets:
        antibody_ids += [re.findall('>Antibody [A-Za-z0-9]*<', str(souplet))[0][10:-1]]

    id_index = np.where([id == antibody_id for id in antibody_ids])[0][0]
    
    start_ind = antibody_html_str.find('Matching transcripts')
    end_ind = antibody_html_str.find('<th class="sub_head"', start_ind)
    
    split = antibody_html_str[start_ind:end_ind].split('<td class="" style="">')[id_index+1]
    
    ensp_soup = BeautifulSoup(split, 'html.parser')
    
    souplets = ensp_soup.find_all('a', attrs={'rel': 'nofollow noopener'})
    
    protein_ids = list()
    for souplet in souplets:
        protein_ids += [re.findall('ENSP[A-Za-z0-9]* ', str(souplet))[0][:-1]]
        
    return protein_ids

def get_urls_and_info_from_xml(xml_tree):
    urls = []
    for proteinAtlas in xml_tree.iter('proteinAtlas'):
        
        identifier = filter_element(filter_element(proteinAtlas, 'entry'), 'identifier')
        ensg_id = identifier.attrib['id'] 
        antibody_html_bin = ensg_to_antibody_html(ensg_id)
        
        for antibody in proteinAtlas.iter('antibody'):
            
            antibody_id = antibody.attrib['id']
            ensp_ids = antibody_and_html_to_ensp(antibody_id, antibody_html_bin)
            
            for data in antibody.iter('data'):
                for image in data.iter('image'):
                    for imageUrl in image.iter('imageUrl'):
                        if 'green' in imageUrl.text:
                            channels = [x for x in list(image) if x.tag == 'channel']
                            channel_info = {channel.attrib['color']:channel.text for channel in channels}
                            antigenSequence = filter_element(antibody,'antigenSequence')
                            cellLine = filter_element(data, 'cellLine')
                            
                            xref = filter_element(identifier, 'xref')
                            urls += [{'antibody':antibody.attrib['id'] if (antibody is not None and antibody.attrib['id'] is not None) else np.nan,
                                      'ENSG':identifier.attrib['id'] if (identifier is not None and identifier.attrib['id'] is not None) else np.nan,
                                      'ENSP':ensp_ids,
                                      'proteinName':xref.attrib['id'] if (xref is not None and xref.attrib['id'] is not None) else np.nan,
                                      'antigenSequence':antigenSequence.text if (antigenSequence is not None and antigenSequence.text is not None) else np.nan,
                                      'cellLine':cellLine.text if (cellLine is not None and cellLine.text is not None) else np.nan,
                                      'imageUrls':get_single_channel_urls(imageUrl.text, channel_map=channel_info) if (imageUrl is not None and imageUrl.text is not None) else np.nan}]
    return urls


def make_local_img_path(url):
    return url.split('images/')[-1]

def get_img_and_write_local_and_return_path(url, parent_dir):
    local_img_path = make_local_img_path(url)
    fpath = os.path.join(parent_dir,local_img_path)
    if not os.path.isfile(fpath):
        req = urllib.request.Request(url)
        img = urllib.request.urlopen(req).read()
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'wb') as handler:
            handler.write(img)
    return(local_img_path)

def ensg_to_info(ensg_id):
    xml_url = 'https://www.proteinatlas.org/{}.xml'.format(ensg_id)
    
    doc = make_xml_tree_from_url(xml_url)
    urls_and_info = get_urls_and_info_from_xml(doc)
        
    return urls_and_info 
#     all_urls_and_info += [*urls_and_info]
    
def ensg_to_disk(ensg_id, save_dir, info_file_name = 'info.csv', overwrite=False):
    save_file = save_dir + '/' + info_file_name
        
    if os.path.exists(save_file) and not overwrite:
        return
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    urls_and_info = ensg_to_info(ensg_id)
    
    df = pd.DataFrame(columns=['ENSG',  'ENSP', 'proteinName', 'antibodyName', 'antigenSequence', 'cellLine',
                           'antibodyChannel', 'microtubuleChannel', 'nuclearChannel', 'segmentationChannel'])
    
    for u in urls_and_info:
        
        antibody_path_local = get_img_and_write_local_and_return_path(u['imageUrls']['singleChannelUrls']['antibody'], save_dir)
        microtubule_path_local = get_img_and_write_local_and_return_path(u['imageUrls']['singleChannelUrls']['microtubules'], save_dir)
        nuclear_path_local = get_img_and_write_local_and_return_path(u['imageUrls']['singleChannelUrls']['nucleus'], save_dir)
        segmentation_path_local = get_img_and_write_local_and_return_path(u['imageUrls']['segmentationUrl'], save_dir)

        df = df.append({'ENSG':u['ENSG'],
                        'ENSP':u['ENSP'],
                        'proteinName':u['proteinName'],
                        'antibodyName':u['antibody'],
                        'antigenSequence':u['antigenSequence'],
                        'cellLine':u['cellLine'],                    
                        'antibodyChannel':antibody_path_local,
                        'microtubuleChannel':microtubule_path_local,
                        'nuclearChannel':nuclear_path_local,
                        'segmentationChannel':segmentation_path_local}, ignore_index=True)

    df.to_csv(save_file)