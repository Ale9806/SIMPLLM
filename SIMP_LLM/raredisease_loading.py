# AUTHORS: Selina (except the print_head function by Alejandro)

import pandas as pd
import numpy as np
import tabulate
import xml.etree.ElementTree as ET
import os
import re



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




######## Functions to identify rare diseases in DRKG ##################

# Helper function to see DRKG ontologies
def get_drkg_entity_ontologies(df, entity_list):
    """
    From DRKG entity dataframe, get all ontology types associated with input entity types
    """
    df['matched'] = np.where((df['name'].isna()) | (df['entity_type'] == 'Tax'), 0, 1) # Flag if matched

    # Count ontologies associated with entities
    ontology_counts = df.loc[df['entity_type'].isin(entity_list)].groupby(['matched', 'ontology_name']).agg(
        count = ('drkg_id', 'count')
    ).reset_index()

    return ontology_counts


def read_and_process_doid(relation_file, filter_regex=r'UMLS|ICD|OMIM|MESH', verbose=False):
  """  
  Process Disease Ontology DOID lookup table in the following ways:
  - Remove extra variables
  - Filter to entries with code types in regex input (choose ontology types in Orphanet, in all caps)
  """
  df_raw = pd.read_csv(relation_file)

  # Subset to relevant variables
  doid_vars = ['id', 'Preferred Label', 'Synonyms', 'Definitions', 'CUI', 'database_cross_reference', 'has_alternative_id', 'has_exact_synonym', 'Parents']
  df_subset = df_raw[doid_vars]

  # Filter DOID to relevant codes
  df_filter = df_subset[df_subset['database_cross_reference'].str.upper().str.contains(filter_regex, na=False)]

  if verbose:
    print(f"\n DOID Dataframe (After processing):\n")
    print_head(df_filter)

  return df_filter



def create_orphanet_regex(orphan_codes, verbose=False):
    """
    Create regex strings for codes in Orphanet to search in DOID mapped codes with the following rules:

    # Code ontology is at beginning of string or immediately after '|'
    # Cannot have '|' between code ontology and code value
    # Have ':' immediately before all code values
    # For ICD codes, need to make sure to escape the middle period
    # For all other codes beside ICD, code value must immediately precede end of string or '|' to allow no extra characters. 
    # However, extra characters are OK for ICD since they indicate a subset of the rare disease

    # Regex format: 
    # ICD:   '(?:^|\|)ICD10(?:(?!\|).)*:E88\.9.*'
    # Other (UMLS as example): '(?:^|\|)UMLS(?:(?!\|).)*:C0033(?:$|\|)'
    """
    orphan_codes_match = orphan_codes.copy()

    # Escape period for ICD
    orphan_codes_match['code'] = np.where(orphan_codes_match['code_source'].str.startswith('ICD'), 
                                        orphan_codes_match['code'].str.replace('.','\.'),
                                        orphan_codes_match['code'])
    
    # Create regex strings to follow above rules
    orphan_codes_match['regex'] = '(?:^|\|)'+orphan_codes['code_source'].str.upper()+'(?:(?!\|).)*:'+orphan_codes['code']
    orphan_codes_match['regex'] = orphan_codes_match['regex'].str.replace('-', '', regex=True)
    orphan_codes_match['regex'] = np.where(orphan_codes_match['code_source'].str.startswith('ICD'), 
                                        orphan_codes_match['regex']+'.*',
                                        orphan_codes_match['regex']+'(?:$|\|)')
    
    if verbose:
        print(f"\n Orphanet Dataframe with regex (After processing):\n")
        print_head(orphan_codes_match)

    return orphan_codes_match


def merge_regex(regex_df, regex_col, search_df, search_col, verbose=False):
    """Get mapping by regex"""
    # Adapted from: https://stackoverflow.com/questions/62521616/can-i-perform-a-left-join-merge-between-two-dataframes-using-regular-expressions
    idx = [(i,j) for i,r in enumerate(regex_df[regex_col]) for j,v in enumerate(search_df[search_col].astype(str)) if re.match(r,v)]
    regex_df_idx, search_df_idx = zip(*idx)
    t = regex_df.iloc[list(regex_df_idx),0].reset_index(drop=True)
    t1 = search_df.iloc[list(search_df_idx),0].reset_index(drop=True)
    outdata = pd.concat([t,t1],axis=1)
    if verbose:
        print(f"\n Regex mapping:\n")
        print_head(outdata)
    return outdata


def check_raredisease_multiple_codes(matched_rarediseases, verbose=False):
    """Check for rare diseases with multiple code types"""
    # DRKG entities with multiple orphanet codes?
    ct_orphacode = matched_rarediseases.groupby(by=['drkg_id','name', 'match_type']).agg(
        ct_orphacode = ('Orphacode', 'count')
    ).reset_index()
    multiple_orphacode = ct_orphacode[ct_orphacode['ct_orphacode']>1]

    # Orphacodes with multiple DRKG entities?
    ct_drkg_id = matched_rarediseases.groupby(by=['Orphacode','name', 'match_type']).agg(
        ct_drkg_id = ('drkg_id', 'count')
    ).reset_index()
    multiple_drkg = ct_drkg_id[ct_drkg_id['ct_drkg_id']>1]

    if verbose:
        print('DRKG IDs associated with multiple Orphacodes: ', multiple_orphacode.shape[0])
        print('Orphacodes associated with multiple DRKG IDs: ', multiple_drkg.shape[0])

    return multiple_orphacode, multiple_drkg


def find_drkg_rarediseases(drkg_all_entities, orphan_codes, orphacode_doid_regex, verbose=False):
    """
    Find DRKG entity IDs that match orphan disease codes from 
    - orphan_codes (original Orphanet codes)
    - orphacode_doid_regex (Orphanet codes matched to external DOID codes)
    """
    doid_orphan_codes = orphacode_doid_regex.merge(orphan_codes, how='left', on='Orphacode').drop_duplicates()

    # Get matches on MeSH and OMIM codes
    orphan_codes['code_source_upper'] = orphan_codes['code_source'].str.upper()
    match_try1 = drkg_all_entities.merge(orphan_codes, how='inner', left_on=['ontology_name', 'code'], right_on=['code_source_upper', 'code'])
    match_try1['match_type'] = 'MeSH/OMIM'

    # Get matches on DOID regex matches with other Orphanet codes
    match_try2 = drkg_all_entities.merge(doid_orphan_codes, how='inner', left_on='ontology_code', right_on='id')
    match_try2['match_type'] = 'DOID regex'

    # Match by name
    match_try3 = drkg_all_entities.merge(orphan_codes, how='inner', left_on=drkg_all_entities['name'].str.upper(), right_on=orphan_codes['Name'].str.upper())
    match_try3['match_type'] = 'Disease name'

    # Stack matched entities
    matched_rarediseases = pd.concat([match_try1, match_try2, match_try3], ignore_index=True, axis=0).drop_duplicates(subset=['drkg_id', 'Orphacode']) # keeps first entry of duplicates

    if verbose:
        print(f"\n DRKG-Orphacode matches:\n")
        print_head(matched_rarediseases)

        matches_by_type = matched_rarediseases.groupby(by=['match_type', 'entity_type', 'ontology_name']).agg(
            count = ('drkg_id', 'count')
        )
        print(matches_by_type)

        print('\nUnique matched rare disease DRKG IDs: ', len(matched_rarediseases['drkg_id'].unique()))
        print('Unique matched Orphacodes: ', len(matched_rarediseases['Orphacode'].unique()))
        _, _ = check_raredisease_multiple_codes(matched_rarediseases, verbose=verbose)

    return matched_rarediseases



######## Functions to match rare diseases and drugs between DRKG and Drug Repurposing Hub database ##################

def read_and_process_rep_drugs(relation_file, verbose=False):
  """  
  Process OMIM lookup table in the following ways:
  - Remove extra variables
  - Clean disease name
  - Add "Disease::OMIM:" in front of OMIM ID to match DRKG format
  """
  df = pd.read_csv(relation_file, sep="\t", comment='!')

  if verbose:
    print(f"\n {relation_file}  Drug Repurposing Dataframe:\n")
    print_head(df)
  return df