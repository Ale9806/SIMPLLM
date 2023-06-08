### AUTHORS: Alejandro and Selina (equal)

import tabulate
import pandas as pd
import numpy as np




######## Read Functions  ##################
def print_head(df:pd.core.frame.DataFrame,n:int=5) -> None:
  print(tabulate.tabulate(df.head(n) , headers='keys', tablefmt='psql'))

### AUTHORS: from DRKG, modularized by Alejandro
def get_triplets(drkg_file:str = 'drkg.tsv',verbose:bool=False) -> list:
  """ Read drkg.tsv file and return triplets """

  df        = pd.read_csv(drkg_file, sep="\t", header=None, engine="pyarrow")
  triplets  = df.values.tolist()

  if verbose: 

    print("\n Triplets:\n")
    print(triplets[0:10])

    print(f"\n {drkg_file}  Dataframe:\n")
    print_head(df)
  return triplets,df



### AUTHORS: Alejandro
def  read_tsv(relation_file:str,verbose:bool=False):
  """ Read glossary """ 
  df = pd.read_csv(relation_file, sep="\t",engine="pyarrow")

  if verbose:
    print(f"\n {relation_file}  Dataframe:\n")
    print_head(df)
  return df





########## Filter & Map Functions ###################
def filter_drkg(data_frame:pd.core.frame.DataFrame, filter_column:int,filter_term:str,verbose:bool=False) -> pd.core.frame.DataFrame:
  """
    Arguments:
      filter_column<int>: column use to filter 
      fitler_term<str>:   string (use for Regex) capturing  either the interaction to filter or head/tail  e.g.: r'.*?Compound:Disease'
 
    Outputs: 
      df:<pd.core.frame.DataFrame> A filter dataframe 
  """
  relations           = pd.Series(data_frame[filter_column].unique())
  relations_filtered  = relations[relations.str.contains(filter_term, regex=True)]

  if verbose: 
    print(f"Number of Rows Before Filtering: {len(relations)}")
    print(f"Number of Rows After Filtering: {len(relations_filtered )}\n")
    print("\nFiltered:")
    print(relations_filtered ) 

  return relations_filtered 




def map_drkg_relationships(df_1,relation_glossary,verbose:bool=False):
  df_1_ = df_1.to_frame().merge( relation_glossary, left_on=0, right_on='Relation-name', how='left')

  if verbose:
    print("\nRelationships Mapped:")
    print(df_1_['Interaction-type'])
  
  return  df_1_



def filter_interaction_subset(df:pd.core.frame.DataFrame,filter_colunm_name:str,regex_string:str,return_colunm_name:str=None) -> pd.core.frame.DataFrame:
    """
    Arguments:
      df:<pd.core.frame.DataFrame>:    DataFrame to filter 
      filter_colunm_name<str>:         Dataframe column use to do the filtering 
      regex_string<str>:               Regular Expression use to filter e.g. 'treat|inhibit|alleviate
      return_colunm_name<str:optional> Optional Name of the filtered column to return, if None it returns the dataframe 


      filter_column<int>: column use to filter 
      fitler_term<str>:   string (use for Regex) capturing  either the interaction to filter or head/tail  e.g.: r'.*?Compound:Disease'
 
    Outputs: 
      df:<pd.core.frame.DataFrame> A filter dataframe 
  """

    subset = df[df[filter_colunm_name].str.contains(regex_string, regex=True)]  # Filter dataframe based on regex

    ### Return a specific column if user requests it ###
    if return_colunm_name != None: 
        subset  =  subset[return_colunm_name]


    return subset



def get_unique_values(df, colunm:int) :
  """  Check if any entries are null or contain :: """
  df0_test = np.unique(df[colunm][df[colunm].str.contains("::")].to_numpy())
  return df0_test


