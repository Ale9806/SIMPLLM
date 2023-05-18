
import pandas as pd
import numpy as np

  
########## DRKG entity and relationship processing ###################
def get_unique_entities(df:pd.core.frame.DataFrame, columns):
  """Append all unique entries in specified list of columns in dataframe and get unique entities
  """
  entity_list = []
  for col in columns:
    entity_list = np.append(entity_list, df[col])
  entity_list = np.unique(entity_list)
  return entity_list


def get_entity_lookup(drkg_entities, code_dict):
    """Converts list of unique DRKG entities to entity table with the following items, using the lookup table dictionary code_dict:
        'drkg_id':       original entity code in DRKG
        'drkg_dict_id':  original entity code, except with entity name in MeSH entity codes removed to match with MeSH lookup format
        'name':          natural language entity name, translated using node_dict dictionary
        'entity_type':   type of entity (gene, disease, compound, etc.), taken from drkg_id
        'ontology_code': combined ontology name and code, taken by removing entity_type from drkg_id
        'ontology_name': name of ontology from which code was sourced, if available
        'code':          specific code or ID from the ontology

    Also does the following cleaning:
    * Manual processing of entity and ontology names where the name or source was inferred from the code/ID
    * Remove irrelevant entries (taxonomy and entries with only an entity type but no associated code such as "Gene::")
    * Remove entities with no name and return them as a separate dataframe
    """
    drkg_entity_df = pd.DataFrame(drkg_entities, columns=['drkg_id'])

    # Create copy of DRKG ID value that simplifies MeSH codes
    drkg_entity_df['drkg_dict_id'] = drkg_entity_df['drkg_id'].str.replace(r'.*?MESH:', "MESH::", regex=True)

    # Map entity natural language name
    drkg_entity_df['name'] = drkg_entity_df['drkg_dict_id'].map(code_dict)

    # Get ontology name and code if available
    drkg_entity_df[['entity_type', 'ontology_code']] = drkg_entity_df['drkg_id'].str.split("::", expand=True)
    drkg_entity_df['ontology_name'] = drkg_entity_df['ontology_code'].str.split(":", n=2, expand=True)[0]
    drkg_entity_df['code'] = drkg_entity_df['ontology_code'].str.split(":", n=2, expand=True)[1]

    ###### Cleaning ######
    # Move codes without ontology names to correct column
    drkg_entity_df.loc[drkg_entity_df['ontology_name'] == drkg_entity_df['ontology_code'], 'ontology_name'] = None
    drkg_entity_df.loc[drkg_entity_df['code'].isna(), 'code'] = drkg_entity_df['ontology_code']

    # Add name for entries with SARS-CoV code
    drkg_entity_df.loc[drkg_entity_df['code'].str.startswith('SARS-CoV2'), 'name'] = drkg_entity_df['code']

    # Manually correct specific ontology names without ':' as ontology-code divider
    drkg_entity_df.loc[drkg_entity_df['ontology_code'].str.startswith('CHEMBL'), 'ontology_name'] = 'CHEMBL'
    drkg_entity_df.loc[drkg_entity_df['entity_type'] == 'Atc', 'ontology_name'] = 'Atc'
    drkg_entity_df.loc[(drkg_entity_df['entity_type'] == 'Compound') & (drkg_entity_df['ontology_code'].str.startswith('DB')), 'ontology_name'] = 'drugbank'
    drkg_entity_df.loc[(drkg_entity_df['entity_type'] == 'Side Effect') & (drkg_entity_df['ontology_code'].str.len() == 8), 'ontology_name'] = 'UMLS CUI'
    drkg_entity_df.loc[(drkg_entity_df['entity_type'] == 'Symptom') & (drkg_entity_df['ontology_code'].str.len() == 7), 'ontology_name'] = 'MESH'

    # Remove entities that are irrelevant or without name (save for downstream analysis)
    drkg_unmatched = drkg_entity_df[(drkg_entity_df['name'].isna()) | 
                                    (drkg_entity_df['entity_type'] == 'Tax') |
                                    (drkg_entity_df['ontology_code'].isna())] # ontology_code filter is redundant to name filter, but keeping in case we need this subset later
    drkg_entity_df = drkg_entity_df[~drkg_entity_df.index.isin(drkg_unmatched.index)]

    return drkg_entity_df, drkg_unmatched 


def OLD_convert_entitynames(df, col, code_dict):
  """Keeping here for reference: Convert entity codes to names in specified column based on dictionary with additional cleaning"""
  df_update = df.copy()
  df_update[col] = df_update[col].str.replace(r'.*?MESH:', "MESH::", regex=True) # Remove MeSH labeling
  df_update[col] = df_update[col].map(code_dict).fillna(df_update[col])    # Translate dictionary
  df_update[col] = df_update[col].str.replace("Gene::", "Gene ID ") # For remaining uncoverted Gene IDs, remove "::"
  df_update[col] = df_update[col].str.replace("Disease::", "") # For remaining diseases (appears to be just SARS-COVID related names), remove label
  return df_update


def convert_entitynames(df, col, node_dict):
  """Convert entity codes to names in specified column based on dictionary"""
  df_update = df.copy()
  df_update[col] = df_update[col].map(node_dict)    # Translate dictionary, dont replace NAs
  return df_update


def flip_headtail(df, search_string):
    """Flip heads and tails where relationship contains certain string"""
    df_update = df.copy()
    heads = df_update[0].copy()
    df_update.loc[df_update[1].str.contains(search_string), 0] = df_update[2]
    df_update.loc[df_update[1].str.contains(search_string), 2] = heads
    return df_update