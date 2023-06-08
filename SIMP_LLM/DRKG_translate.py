
### AUTHORS: Selina primary, Alejandro and Alex where specified

import tabulate
import pandas as pd
import chembl_downloader
import os 
import re
import sys


########## Lookup Table Processing Functions ###################
# AUTHORS: Alejandro
def print_head(df:pd.core.frame.DataFrame,n:int=5) -> None:
  print(tabulate.tabulate(df.head(n) , headers='keys', tablefmt='psql'))

  


# AUTHORS: Alejandro
def  read_tsv(relation_file:str,verbose:bool=False, **kwargs):
  """ Read glossary """ 
  df = pd.read_csv(relation_file, sep="\t",engine="pyarrow", **kwargs)

  if verbose:
    print(f"\n {relation_file}  Dataframe:\n")
    print_head(df)
  return df




# AUTHORS: Selina, modularized by Alejandro
def process_hetionet(df, verbose=False):
  """  
  Process entity names in Hetionet lookup table for clarity in the following ways:
  - Add "gene" after gene types
  """

  temp = df.copy()
  temp["name"] = temp['name'].astype(str) + ' gene'

  df_updated = df.copy()
  df_updated.loc[df["kind"] == "Gene", "name"] = temp["name"]
  
  if verbose:
    het_sample = df.drop_duplicates(subset=['kind'])
    print(f"\n Sample of Hetionet Data Types (Before processing):\n")
    print_head(het_sample, n=het_sample.shape[0])

    het_sample = df_updated.drop_duplicates(subset=['kind'])
    print(f"\n Sample of Hetionet Data Types (After processing):\n")
    print_head(het_sample, n=het_sample.shape[0])

  return df_updated


# AUTHORS: Selina, one line edit by Alex to correct variable name for new source file
def read_and_process_gene_ID(relation_file, verbose=False):
  """
  Written by Alex Loia 
  Process Gene ID lookup table in the following ways:
  - Filter to type = symbol (exclude synonym duplicates)
  - Add "Gene::" in front of Gene ID to match DRKG format
  - Add "gene" after gene types
  """
  df = read_tsv(relation_file, usecols=['GeneID', 'description', 'Symbol'])
  df_updated = df.rename(columns={"Symbol": "symbol"})
  df_updated['description'] = df_updated['description'].astype(
      str) + (' (' + df_updated['symbol'].astype(str) + ')')
  df_updated['GeneID'] = "Gene::" + df_updated['GeneID'].astype(str)
  df_updated = df_updated.drop(columns=['symbol'])

  if verbose:
    print(f"\n Gene ID Dataframe (After processing):\n")
    print_head(df_updated)

  return df_updated


# AUTHORS: Selina
def read_and_process_drugbank(relation_file, verbose=False):
  """  
  Process DrugBank lookup table in the following ways:
  - Remove extra variables
  - Add "Compound::" in front of DrugBank ID to match DRKG format
  """
  df = pd.read_csv(relation_file)

  df_updated = df[['DrugBank ID', 'Common name']].copy()
  df_updated['DrugBank ID'] = "Compound::" + df_updated['DrugBank ID'].astype(str)

  if verbose:
    print(f"\n DrugBank Dataframe (Before processing):\n")
    print_head(df)

    print(f"\n DrugBank Dataframe (After processing):\n")
    print_head(df_updated)

  return df_updated


# AUTHORS: Selina
def read_and_process_omim(relation_file, verbose=False):
  """  
  Process OMIM lookup table in the following ways:
  - Remove extra variables
  - Clean disease name
  - Add "Disease::OMIM:" in front of OMIM ID to match DRKG format
  """
  df_raw = pd.read_csv(relation_file, sep="\t", skiprows=2)

  df = df_raw[['MIM Number', 'Preferred Title; symbol']].dropna()
  df['MIM Number'] = df['MIM Number'].astype(int)
  df[['name', 'symbol', 'extra']] = df['Preferred Title; symbol'].str.split(';', expand=True)
  df = df[['MIM Number', 'name']]
  df['MIM Number'] = "Disease::OMIM:" + df['MIM Number'].astype(str)

  if verbose:
    print(f"\n {relation_file}  Dataframe (Before processing):\n")
    print_head(df)

    print(f"\n {relation_file}  Dataframe (After processing):\n")
    print_head(df)
  return df



# AUTHORS: Selina, Adapted from: https://code.tutsplus.com/tutorials/working-with-mesh-files-in-python-linking-terms-and-numbers--cms-28587b
def create_mesh_dict(relation_files=['c2023.bin', 'd2023.bin'], verbose=False):
  """  
  Process MeSH terms
  Input: List of .bin filenames
  Output: Dictionary of MeSH IDs and names
  - Add "MESH::" in front of all entries to easily check later if terms in DRKG were not converted
  """

  # Adapted from: https://code.tutsplus.com/tutorials/working-with-mesh-files-in-python-linking-terms-and-numbers--cms-28587b
  mesh_ids = {}
  for meshFile in relation_files:
    with open(meshFile, mode='rb') as file:
        mesh = file.readlines()

    if meshFile.startswith('c'):
        reg_val = b'NM = (.+)$'
    else:
        reg_val = b'MH = (.+)$'

    # Create dictionary of MeSH entity names with their UIDs
    term = None
    for line in mesh:
        meshTerm = re.search(reg_val, line)
        if meshTerm:
            term = meshTerm.group(1)
        meshNumber = re.search(b'UI = (.+)$', line)
        if meshNumber:
            if term is not None:
              number = meshNumber.group(1)
              mesh_ids['MESH::' + number.decode('utf-8')] = term.decode('utf-8')

  if verbose:
      print(f"\n MeSH Dictionary:\n")
      print_head(pd.DataFrame.from_dict(mesh_ids, orient='index'))

  return mesh_ids

# AUTHORS: Selina
def download_chembl():
  # Import ChEMBL data
  sql = """
  SELECT DISTINCT
      MOLECULE_DICTIONARY.chembl_id,
      MOLECULE_DICTIONARY.pref_name,
      MOLECULE_DICTIONARY.chebi_par_id
      FROM MOLECULE_DICTIONARY
      WHERE MOLECULE_DICTIONARY.pref_name IS NOT NULL OR MOLECULE_DICTIONARY.chebi_par_id IS NOT NULL
  """
  df = chembl_downloader.query(sql)
  return df




# AUTHORS: Selina
def process_chebi_chembl(chebi_file, chembl_df_raw, verbose=False): 
  """  
  Process and download ChEBI and ChEMBL lookup tables
  - Add "Compound::" to names
  """

  # Read ChEBI file
  chebi_df_raw = pd.read_csv(chebi_file, sep='\t', compression='gzip')
  chebi_df = chebi_df_raw[['CHEBI_ACCESSION', 'NAME']].dropna()
  chebi_df['CHEBI_ACCESSION'] = "Compound::" + chebi_df['CHEBI_ACCESSION'].astype(str)

  # Fill missing CHEMBL names with CHEBI name where possible
  chembl_chebi_combined = chembl_df_raw.merge(chebi_df_raw[['ID', 'NAME']], how='left', left_on='chebi_par_id', right_on='ID')
  chembl_df = chembl_chebi_combined.copy()
  chembl_df.loc[chembl_chebi_combined["pref_name"].isna(), "pref_name"] = chembl_chebi_combined["NAME"]
  chembl_df = chembl_df[['chembl_id', 'pref_name']].dropna()
  chembl_df['chembl_id'] = "Compound::" + chembl_df['chembl_id'].astype(str)

  return chebi_df, chembl_df





# AUTHORS: Alejandro created this modularized code, Selina found file names, Alex updated gene_df file name
def load_lookups(data_path,verbose=False):
    hetionet_df_raw   =  read_tsv(os.path.join(data_path,'hetionet-v1.0-nodes.tsv'),verbose=verbose)    # Read relationship mapping
    hetionet_df       =  process_hetionet(df=hetionet_df_raw, verbose=verbose)    # Process entity names for clarity (e.g., F8 -> Gene F8) 

    gene_df           =  read_and_process_gene_ID(os.path.join(data_path, "Homo_sapiens.gene_info"), verbose=verbose)                 

    drugbank_df       =  read_and_process_drugbank(os.path.join(data_path,'drugbank vocabulary.csv'), verbose=verbose) # Read and process DrugBank IDs

    omim_df           =  read_and_process_omim(os.path.join(data_path,'mimTitles.txt'), verbose=verbose) # Read and process OMIM disease IDs

    mesh_dict         =  create_mesh_dict(relation_files=[os.path.join(data_path,'c2023.bin'), os.path.join(data_path,'d2023.bin')], verbose=verbose) # Create dictionary for MeSH terms

    chembl_df_raw       = download_chembl() # Import raw ChEMBL data
    chebi_df, chembl_df = process_chebi_chembl(os.path.join(data_path,'compounds.tsv.gz'), chembl_df_raw, verbose=verbose) # Import and process ChEBI and ChEMBL molecule databases

    return hetionet_df, gene_df, drugbank_df, omim_df, mesh_dict, chebi_df, chembl_df

if __name__ == "__main__":
   data_path = sys.argv[1] if len(sys.argv) > 1 else "data/"
   dfs = load_lookups(data_path)
   for df in dfs:
      print(df.head(5))

