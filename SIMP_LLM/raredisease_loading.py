import pandas as pd
import numpy as np
import tabulate
import xml.etree.ElementTree as ET
import os



######## Helper Functions  ##################
def print_head(df:pd.core.frame.DataFrame,n:int=5) -> None:
  print(tabulate.tabulate(df.head(n) , headers='keys', tablefmt='psql'))




######## Functions to read in OrphaNet XML file for orphan disease MeSH and other codes ##################

def printRecur(root, data, cols, ignoreElems=[], passElems=[], appendElems=[], attribElems=[], endElems=[]):
# def printRecur(root):
    """
    Recursively adds elements to data (data) and column name (cols) lists from inputted root of XML file with special element cases:
    * ignoreElems:      list of elements to completely ignore (including their children)
    * passElems:        elements whose children to explore but whose names/text are not to be stored
    * appendElems:      elements to append as a single string instead of storing as separate rows
    * attribElems:      elements whose attributes to store instead of text
    * endElems:         elements whose end should be added as a separate row for future use
    * All other elements will have element name and text stored in 'cols' and 'data' lists, respectively
    """
    for i, child in enumerate(root):
        # Fully ignore some elements and their children
        if child.tag in ignoreElems:            
            continue

        # Look at child elements and add to list unless specified
        if child.tag.title() not in passElems:  
            if child.tag.title() in appendElems and i>0:
                data[-1] = data[-1] + '|' + child.attrib.get('name', child.text)
            else:
                cols.append(child.tag.title())
                if child.tag in attribElems:
                    data.append(list(child.attrib.values())[0])
                else:
                    data.append(child.attrib.get('name', child.text))

        # Look at children of child element
        printRecur(child, data, cols, ignoreElems, passElems, appendElems, attribElems, endElems)

    # Mark end of specified sections for later use
    if root.tag.title() in endElems:            
            cols.append('END_' + root.tag.title())
            data.append('\n')



def clean_long_data(data, cols, verbose=False):
    """
    Clean long-form orphan disease data from binding main data and column output from printRecur()
    """
    # Bind column and data information
    long_df = pd.DataFrame([])
    long_df['cols'] = cols
    long_df['data'] = data

    # Clean long form orphan disease data
    long_df_processed = long_df.copy()
    long_df_processed = long_df_processed.dropna()
    disease_id = 'Orphacode'

    # Flag disease ID
    long_df_processed['disease_id'] = np.where(long_df_processed['cols'] == disease_id, long_df_processed['data'], None)
    long_df_processed['disease_id'] = long_df_processed['disease_id'].ffill()

    # Add code source
    long_df_processed['code_source'] = np.where(long_df_processed['cols'] == 'Source', long_df_processed['data'], None)
    long_df_processed['code_source'] = np.where(long_df_processed['cols'] == 'END_Externalreferencelist', 'SKIP', long_df_processed['code_source'])
    long_df_processed['code_source'] = long_df_processed['code_source'].ffill()

    # Add code
    long_df_processed['code'] = np.where(long_df_processed['cols'] == 'Reference', long_df_processed['data'], None)
    long_df_processed['code'] = np.where(long_df_processed['cols'] == 'END_Externalreferencelist', 'SKIP', long_df_processed['code'])
    long_df_processed['code'] = long_df_processed['code'].ffill()

    # Rename 'Name' rows with true name 1 row up
    long_df_processed['cols'] = np.where((long_df_processed['cols'] == 'Name') & (long_df_processed['data'].shift(1).str.startswith('\n')), long_df_processed['cols'].shift(1), long_df_processed['cols'])

    # Remove \n rows
    long_df_processed = long_df_processed[~long_df_processed['data'].str.startswith('\n')]

    # Rename cols associated with specific source and remove source columns
    long_df_processed = long_df_processed[~long_df_processed['cols'].str.contains('Source')]

    # Manually consolidate 'definition' entries
    if long_df_processed[long_df_processed['cols']=='Textsectiontype'].drop_duplicates(subset='data').shape[0] == 1:
        long_df_processed['cols'] = np.where(long_df_processed['cols'] == 'Contents', 'Definition', long_df_processed['cols'])
        long_df_processed = long_df_processed[long_df_processed['cols'] != 'Textsectiontype']

    if verbose:
        print(f"\n Long-form orphan disease data (before processing):\n")
        print_head(long_df)
        print(f"\n Long-form orphan disease data (after processing):\n")
        print_head(long_df_processed)

    return long_df_processed


def pivot_orphan_data(long_df_processed, verbose=False):
    """
    Pivot long-form orphan disease data to 2 wide-form datasets: orphan disease names and codes
    """

    #################################################################
    # Get names and descriptions for each orphan disease
    #################################################################

    orphan_names = long_df_processed[long_df_processed['code_source'].isin([None, 'SKIP'])]
    colnames = orphan_names['cols'].drop_duplicates().to_list()
    orphan_names = pd.pivot(orphan_names,  index='disease_id', columns='cols', values='data').reindex(colnames, axis=1)

    #################################################################
    # Get codes for each orphan disease
    #################################################################

    # Merge orphan disease names and Orphacodes
    _orphan_codes = long_df_processed[~long_df_processed['code_source'].isin([None, 'SKIP'])].merge(orphan_names[['Orphacode', 'Name']], how='left', left_on='disease_id', right_on='Orphacode')

    # Create disease-code index for pivoting
    _orphan_codes['id_code'] = _orphan_codes['disease_id'] + _orphan_codes['code_source'] + _orphan_codes['code']

    # Get order of column names to keep this original column order after pivoting
    colnames = _orphan_codes['cols'].drop_duplicates().to_list()

    # Pivot to get info for each disease-code combination
    orphan_codes = pd.pivot(_orphan_codes, index='id_code', columns='cols', values='data').reindex(colnames, axis=1).reset_index()

    # Final column reordering and cleaning
    col_list = ['Orphacode', 'Name', 'code_source', 'id_code']
    orphan_codes = _orphan_codes[col_list].drop_duplicates().merge(orphan_codes, how='left', on='id_code')
    orphan_codes = orphan_codes.drop(columns=['id_code']).rename(columns={'Reference':'code'})

    if verbose:
        print(f"\n Orphan disease name/summary data:\n")
        print_head(orphan_names)
        print(f"\n Orphan disease codes:\n")
        print_head(orphan_codes)

    return orphan_names, orphan_codes




def get_orphan_data(relation_file = 'en_product1-Orphadata.xml', verbose=False):
    """
    Get relevant orphan disease information from Orphanet XML file
    """
    if os.path.isfile(relation_file) == False:
        print('Orphanet file not found in this directory. May need to download from Google Drive data folder.')

    tree = ET.parse(relation_file)
    root = tree.getroot()[1]

    ignoreElems = ['DisorderFlagList', 'DisorderType', 'DisorderGroup','DisorderDisorderAssociationList']
    passElems = ['Disorder', 'Expertlink', 'Synonymlist', 'Externalreferencelist', 'Externalreference']
    attribElems = []
    appendElems = ['Synonym']
    endElems = ['Externalreferencelist']

    data = []
    cols = []

    printRecur(root, data, cols, ignoreElems, passElems, appendElems, attribElems, endElems)
    # printRecur(root)
    long_df_processed = clean_long_data(data, cols, verbose=verbose)
    orphan_names, orphan_codes = pivot_orphan_data(long_df_processed, verbose=verbose)

    return orphan_names, orphan_codes