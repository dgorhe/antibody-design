# from bs4 import BeautifulSoup as soup
import re
import pprint
import wget
import git 
import pathlib
import os
from tqdm import tqdm

git_repo = git.Repo(".", search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
pages = pathlib.Path(os.path.join(git_root, "data", "pages"))

test = """
var receptorTableData = {"header":"","data":{"receptorsData":[{"chain1v":"TRAV26-1*01","chain1d":"","chain1j":"TRAJ37*01","chain1cdr1":"TISGNEY","chain1cdr2":"GLKNN","chain1cdr3":"IVVRSSNTGKLI","chain1vdomseq":"TTQPPSMDCAEGRAANLPCNHSTISGNEYVYWYRQIHSQGPQYIIHGLKNNETNEMASLIITEDRKSSTLILPHATLRDTAVYYCIVVRSSNTGKLIFGQGTTLQVKP","chain2v":"TRBV14*01","chain2d":"","chain2j":"TRBJ2-3*01","chain2cdr1":"SGHDN","chain2cdr2":"FVKESK","chain2cdr3":"ASSQDRDTQY","chain2vdomseq":"GVTQFPSHSVIEKGQTVTLRCDPISGHDNLYWYRRVMGKEIKFLLHFVKESKQDESGMPNNRFLAERTGGTYSTLKVQPAELEDSGVYFCASSQDRDTQYFGPGTRLTVL","epitopes":"VMAPRTLIL (<iedb_link_epitope69921>)","linksDataString":[{"id":"epitope69921","text":"2","target":"internal","type":"recepter_epitope_assays","parameters":{"epitope_id":"69921","assay_ids":["1548960","1583178"]}}]}],"chain1Type":"alpha","chain2Type":"beta"}};
"""

keys = {
    "alpha": {
        "v_domain": "chain1vdomseq",
        "cdr1": "chain1cdr1",
        "cdr2": "chain1cdr2",
        "cdr3": "chain1cdr3",
    },
    "beta": {
        "v_domain": "chain2domseq",
        "cdr1": "chain2cdr1",
        "cdr2": "chain2cdr2",
        "cdr3": "chain2cdr3",
    }
}

def get_value(input_string, key):
    # Lookbehind for quotation mark + key + lookhead for quotes, colon, and string of interest
    regex = r"(?<=\"" + key + "\"\:\")\w*(?=\")"
    regexpression = re.compile(regex)
    out = regexpression.findall(input_string)
    
    try:
        assert len(out) == 1
    except AssertionError:
        print("Output from regex found multiple entries, check output")
        pprint(out)
    
    return out.pop()

def scrape_page(path):
    pass

def download_pages(urls, prefix='tcr', dir=pages):
    for url in tqdm(urls):
        filename = os.path.join(dir, f"{prefix}-{url.split('/')[-1]}.txt")
        wget.download(url, filename)


if __name__ == "__main__":
    # print(get_value(test, keys["alpha"]["cdr1"]))
    url_root = "https://www.iedb.org/receptor/"
    epitope_ids = [47, 54, 97, 109]
    urls = [url_root + str(id) for id in epitope_ids]

    download_pages(urls, prefix='tcr', dir=pages)

