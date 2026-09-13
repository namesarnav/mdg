from bs4 import BeautifulSoup
import os
import html
import json
import re
import argparse
from tqdm import tqdm

# INPUT_DIR   = '../..data/timebank_1_2/data/timeml/'
# OUTPUT_DIR  = '../data/timebank_1_2/data/timeml_processed/'

# INPUT_DIR   = '../../data/TempEval-3/DATA-PUBLISHED/TE3-Silver-data/'
# OUTPUT_DIR  = '../../data/TempEval-3/DATA-PUBLISHED/TE3-Silver-data-processed/'

INPUT_DIR   = '../../data/TempEval-3/DATA-PUBLISHED/te3-platinum'
OUTPUT_DIR  = '../../data/TempEval-3/DATA-PUBLISHED/te3-platinum-processed/'

def post_process(_tml_content: str) -> str:
    """
    if the tml_content comes from tempeval3, remove certain tags. Eg.
    **input**:
    <?xml version="1.0" ?>
    <TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd">

    <DOCID>AFP_ENG_19970401.0092</DOCID>

    <DCT>MOSCOW, <TIMEX3 tid="t0" type="TIME" value="1997-04-01" temporalFunction="false" functionInDocument="CREATION_TIME">April 1 , 1997</TIMEX3> (AFP)</DCT>

    <TITLE>Former Russian army chief under corruption probe</TITLE>


    <TEXT>

    Russian prosecutors have <EVENT class="OCCURRENCE" eid="e1">opened</EVENT> a corruption probe against a former commander-in-chief of the army, the Interfax agency <EVENT class="REPORTING" eid="e3">reported</EVENT> <TIMEX3 type="DATE" value="1997-04-01" tid="t1">Tuesday</TIMEX3>.
    ..

    **output**
    <?xml version="1.0" ?>
    <TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd">

    AFP_ENG_19970401.0092

    MOSCOW, <TIMEX3 tid="t0" type="TIME" value="1997-04-01" temporalFunction="false" functionInDocument="CREATION_TIME">April 1 , 1997</TIMEX3> (AFP)

    Former Russian army chief under corruption probe

    Russian prosecutors have <EVENT class="OCCURRENCE" eid="e1">opened</EVENT> a corruption probe against a former commander-in-chief of the army, the Interfax agency <EVENT class="REPORTING" eid="e3">reported</EVENT> <TIMEX3 type="DATE" value="1997-04-01" tid="t1">Tuesday</TIMEX3>.
    ...
    :param _tml_content:
    :return:
    """
    for _tag in ["DOCID", "DCT", "TITLE", "TEXT"]:
        _tml_content = _tml_content.replace(f"<{_tag}>", "").replace(f"</{_tag}>", "")
    _tml_content = html.unescape(_tml_content)  # unescape special chars because that seemed to create problems
    return _tml_content

def process_timebank(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    bad_files = []

    for file_name in tqdm(files):
        if not file_name.endswith(".tml"):
            continue
        curr_data   = {}
        tml_file    = f'{input_dir}/{file_name}'

        with open(tml_file, "r", encoding="utf-8") as file:
            tml_content = post_process(file.read()).strip()
        curr_data['file_name']          = file_name
        
        # Parse the content using BeautifulSoup with the XML parser
        soup = BeautifulSoup(tml_content, "xml")

        makeinstance_tags = soup.find_all('MAKEINSTANCE')
        tlink_tags       = soup.find_all('TLINK')
        slink_tags       = soup.find_all('SLINK')
        alink_tags       = soup.find_all('ALINK')

        tags             = makeinstance_tags + tlink_tags + slink_tags + alink_tags

        # remove these tags from the soup object
        for tag in tags:
            tag.decompose()

        # Now, soup contains the text without the specified tags
        tml_content = soup.find('TimeML').decode_contents().strip()
        tag_data    = []
        tag_pattern = re.compile(r"<(?P<tag>\w+)(?P<attrs>[^>]*)>(?P<text>.*?)</\1>",re.DOTALL)

        tag_spans   = []
        offset      = 0

        # Find all tags with their attributes and text
        # This regex captures the tag name, attributes, and text content
        for match in tag_pattern.finditer(tml_content):
            tag_name    = match.group("tag")
            attr_string = match.group("attrs")
            text        = match.group("text")
            start, end = match.span()

            # Use BeautifulSoup to parse attributes safely
            tag_soup = BeautifulSoup(f"<{tag_name}{attr_string}/>","xml").find(tag_name)
            attrs = dict(tag_soup.attrs) if tag_soup else {}

            tag_spans.append({
                "tag": tag_name,
                "attributes": attrs,
                "text": text,
                "start": start,
                "end": end,
                "start_char": start - offset,
                "end_char": start + len(text) - offset
            })

            offset += end - start - len(text)
            # print(f"Found tag: {tag_name}, Attributes: {attr_string}, Text: {text}, Start: {start}, End: {end}")

        doc_text = soup.get_text().strip()

        curr_data['doc_text'] = doc_text
        curr_data['tags']     = {}

        for tag in tag_spans:
            tag_start_char, tag_end_char, tag_text = tag['start_char'], tag['end_char'], tag['text']
            try:
                assert tag_text == doc_text[tag_start_char:tag_end_char], (f"Text mismatch for tag {tag['tag']}: "
                                                                           f"[tag_text]: {tag_text}, [doc_text]: {doc_text[tag_start_char:tag_end_char]}")
            except AssertionError as e:
                bad_files.append(file_name)
                print(f"AssertionError: {e}")
                print(tml_file)
                print(f"Error: {e}")
                print(f"Text: {tag_text}")
                print(f"Tag: {tag}")
                continue
        
            id_ = None

            for attr in tag['attributes']:
                if attr.endswith('id'):
                    id_ = tag['attributes'][attr]
                    break
            
            curr_data['tags'][id_] = tag
        
        ### save all the tags in the current data document

        # extract all instances of <MAKEINSTANCE> tags, and LINK tags
        ## redoing the parsing to extract the event ids and eiids

        with open(tml_file, "r", encoding="utf-8") as file:
            tml_content = file.read()
        
        # Parse the content using BeautifulSoup with the XML parser
        soup = BeautifulSoup(tml_content, "xml")

        makeinstance_tags = soup.find_all('MAKEINSTANCE')
        tlink_tags        = soup.find_all('TLINK')
        slink_tags        = soup.find_all('SLINK')
        alink_tags        = soup.find_all('ALINK')

        ### we now need to process the document text to form a mapping of the eids with the text spans

        curr_data['event_tags']     = []
        curr_data['eiid2eventid']   = {}

        for mtag in makeinstance_tags:
            # <MAKEINSTANCE eventID="e38" eiid="ei203" tense="PAST" aspect="NONE" polarity="POS" pos="VERB"/>
            # <MAKEINSTANCE eventID="e45" eiid="ei216" tense="NONE" aspect="NONE" polarity="POS" pos="NOUN"/>
            # <MAKEINSTANCE eventID="e26" eiid="ei222" tense="NONE" aspect="NONE" polarity="POS" pos="NOUN"/>
            # <MAKEINSTANCE eventID="e23" eiid="ei218" tense="PAST" aspect="NONE" polarity="POS" pos="VERB"/>

            curr_data['event_tags'].append(mtag.attrs)

            event_id = mtag.get('eventID')
            eiid     = mtag.get('eiid')

            curr_data['eiid2eventid'][eiid] = event_id
        
        curr_data['tlinks'] = []
        for ttag in tlink_tags:
            curr_data['tlinks'].append(ttag.attrs)
        
        curr_data['slinks'] = []
        for stag in slink_tags:
            curr_data['slinks'].append(stag.attrs)
        
        curr_data['alinks'] = []
        for atag in alink_tags:
            curr_data['alinks'].append(atag.attrs)


        with open(f'{output_dir}/{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(curr_data, f, indent=4)
               

    bad_files = list(set(bad_files))

    for bad_file in bad_files:
        print(f'Bad file: {bad_file}')



        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process data for the MDG project.')

    parser.add_argument('--reasoning_type', type=str, default='temporal', help='Reasoning Path')
    parser.add_argument('--dataset',        type=str, default='timebank', help='Path to the input file containing the data.')
    parser.add_argument('--step',           type=str, required=True, help='Process step to execute (e.g., "process", "filter", "split").')


    args = parser.parse_args()


    if args.step == 'process':

        print(f'Processing data for {args.dataset} with reasoning type {args.reasoning_type}...')
        
        if args.dataset == 'timebank' and args.reasoning_type == 'temporal':
            process_timebank(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
          
