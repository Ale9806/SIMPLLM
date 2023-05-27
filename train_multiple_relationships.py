#!/usr/bin/env python
# coding: utf-8

# DRKG
# 
# Adapted from: https://github.com/gnn4dr/DRKG/blob/master/drkg_with_dgl/loading_drkg_in_dgl.ipynb

# In[1]:


import pandas as pd
import numpy as np
import os 
import torch_geometric.transforms as T


# In[2]:


from SIMP_LLM.DRKG_loading   import  get_triplets, read_tsv,filter_drkg,map_drkg_relationships,filter_interaction_subset,print_head
from SIMP_LLM.DRKG_translate import  load_lookups
from SIMP_LLM.DRKG_entity_processing import get_unique_entities, get_entity_lookup, convert_entitynames, flip_headtail
from SIMP_LLM.raredisease_loading import get_orphan_data


# # 1) Load Data

# In[3]:


### 1) Read: This section reads DRKG and a glossary (used to map entities from codes to words)
DATA_DIR           = os.path.join("data")
verbose            =  True 
triplets,drkg_df   =  get_triplets(drkg_file = os.path.join(DATA_DIR  ,'drkg.tsv'),             verbose=verbose)  # Read triplets (head,relationship,tail)
relation_glossary  =  read_tsv(relation_file = os.path.join(DATA_DIR  ,'relation_glossary.tsv'),verbose=verbose)  # Read relationship mapping  


### 2) Filter & Map Interactions: This section returns a list of interactions e.g. DRUGBANK::treats::Compound:Disease )
# 2.1: First  we filter the interactions to only Compound-Disease
# 2.2: Then   we map the codes -> text  (this will be use to further filter interactions based on text) e.g.  Hetionet::CpD::Compound:Disease -> palliation
# 2.3: We use natural text to fitler  interactions based on terms such as "treat" (but we return the orignal interaction name )



# modularize this in create_dataframe
drkg_rx_dx_relations        = filter_drkg(data_frame = drkg_df ,  filter_column = 1 ,  filter_term = r'.*?Compound:Disease', verbose = verbose) # 2.1 Filter only Compound-Disease Interactions
drkg_rx_dx_relations_mapped = map_drkg_relationships(drkg_rx_dx_relations,relation_glossary,verbose=verbose)                                    # 2.2 Map codes to text 

### 2.3 Filter Drug interactions Interaction types to only include: treat inhibit or alleviate interactions  ###
drkg_rx_dx_relation_subset =  filter_interaction_subset(df                  = drkg_rx_dx_relations_mapped,
                                                        filter_colunm_name = 'Interaction-type' ,
                                                        regex_string       =  'treat|inhibit|alleviate',
                                                        return_colunm_name =  'Relation-name')

# 3) Use Filter Interactions to get Gilter DRKG 
drkg_df_filtered = drkg_df[drkg_df[1].isin(drkg_rx_dx_relation_subset)] # 3.1 Filter DRKG  to only  Compund-Disease 
print_head(df=drkg_df_filtered)



###

rx_dx_triplets   = drkg_df_filtered.values.tolist()                     # 3.2 Convert filtered DRKG to list


# In[4]:


# 4) Load Data frames for translation
hetionet_df, gene_df, drugbank_df, omim_df, mesh_dict, chebi_df, chembl_df = load_lookups(data_path=DATA_DIR,verbose=verbose)


# In[5]:


# Load orphan disease names and codes (28 Nov 2022 version)
orphan_names, orphan_codes = get_orphan_data(os.path.join(DATA_DIR, 'en_product1-Orphadata.xml'), verbose=verbose)

# Get orphan disease MeSH codes
orphan_codes_mesh = orphan_codes[orphan_codes['code_source']=='MeSH'].copy()
orphan_codes_mesh['id'] = 'MESH::'+orphan_codes_mesh['code']


# In[6]:


# Make dictionaries for codes
code_df   = pd.concat([hetionet_df[['name', 'id']], 
                       gene_df.rename(columns = {"description":"name", "GeneID":"id"}),
                       drugbank_df.rename(columns = {"Common name":"name", "DrugBank ID":"id"}),
                       omim_df.rename(columns = {"MIM Number":"id"}),
                       chebi_df.rename(columns = {"NAME":"name", "CHEBI_ACCESSION":"id"}),
                       chembl_df.rename(columns = {"pref_name":"name", "chembl_id":"id"}),
                       orphan_codes_mesh.rename(columns = {"Name":"name"}) # SP 05/24/23 added orphan disease MeSH terms
                       ], ignore_index=True, axis=0).drop_duplicates() 
code_dict = pd.Series(code_df['name'].values, index=code_df['id']).to_dict() | mesh_dict # Convert node df to dict and merge with MeSH dictionary

# Get unique DRKG entities
drkg_entities = get_unique_entities(drkg_df, [0,2])

# Create and use convert_entitynames function
drkg_entity_df, drkg_unmatched = get_entity_lookup(drkg_entities, code_dict)

# Create final node dictionary
node_dict = pd.Series(drkg_entity_df['name'].values, index=drkg_entity_df['drkg_id']).to_dict() 

# Initialize translated DRKG and manually clean heads/tails for one case where they were flipped
drkg_translated    = drkg_df.copy()
drkg_translated = flip_headtail(drkg_translated, 'Gene:Compound')

# Map DRKG to translated entity names
drkg_translated = convert_entitynames(drkg_translated, 0, node_dict)
drkg_translated = convert_entitynames(drkg_translated, 2, node_dict)
drkg_translated = drkg_translated.dropna()
print_head(drkg_translated) 

# Summarize percentage translated
print("Number of unique DRKG entities: ", len(drkg_entities)) # should be 97238
print("Number of translated entities: ", drkg_entity_df.shape[0])
print("Number of untranslated entities: ", drkg_unmatched.shape[0])
pct_entity_translated = drkg_entity_df.shape[0]/len(drkg_entities)
print('Percentage of entities translated: ', round(pct_entity_translated*100,1), '%')

print('Total DRKG relationships: ', drkg_df.shape[0])
print('Translated DRKG relationships: ', drkg_translated.shape[0])
pct_translated = drkg_translated.shape[0]/drkg_df.shape[0]
print('Percentage of relationships fully translated: ', round(pct_translated*100,1), '%')


# In[7]:


# Update relation glossary 
relation_df = relation_glossary.copy().rename(columns={'Relation-name':'drkg_id'})
relation_df[['head_entity','tail_entity']] = relation_df['drkg_id'].str.split('::', expand=True)[2].str.split(':', expand=True) # Set head and tail nodes

# Manually fix head and tail nodes for DGIDB relations, which reverse compound-gene interactions
relation_df.loc[relation_df['drkg_id'].str.contains('Gene:Compound'),'head_entity'] = 'Compound'
relation_df.loc[relation_df['drkg_id'].str.contains('Gene:Compound'),'tail_entity'] = 'Gene'

# Fix bioarx entries without the second "::" delimiter
bioarx_ht = relation_df['drkg_id'].str.split(':', expand=True)[[3,4]]
relation_df['head_entity'] = np.where(relation_df['head_entity'].isna(), bioarx_ht[3], relation_df['head_entity'])
relation_df['tail_entity'] = np.where(relation_df['tail_entity'].isna(), bioarx_ht[4], relation_df['tail_entity'])

# Add mapped relation group labels
relation_groups = [['activation', 'agonism', 'agonism, activation', 'activates, stimulates'],
    ['antagonism', 'blocking', 'antagonism, blocking'],
    ['binding', 'binding, ligand (esp. receptors)'],
    ['blocking', 'channel blocking'],
    ['inhibition', 'inhibits cell growth (esp. cancers)', 'inhibits'],
    ['enzyme', 'enzyme activity'],
    ['upregulation', 'increases expression/production'],
    ['downregulation', 'decreases expression/production'],
    ['Compound treats the disease', 'treatment/therapy (including investigatory)', 'treatment']]

relation_df['relation_name'] = relation_df['Interaction-type']

for grp in relation_groups:
    relation_df_subset = relation_df[relation_df['Interaction-type'].isin(grp)].copy()
    for entities in relation_df_subset['Connected entity-types'].unique():
        subgrp = relation_df_subset[relation_df_subset['Connected entity-types'] == entities]['Interaction-type'].unique()
        relation_df.loc[(relation_df_subset['Connected entity-types'] == entities) & (relation_df['Interaction-type'].isin(subgrp)), 'relation_name'] = subgrp[0]

# Remove special characters from relation names
relation_df['relation_name'] = relation_df['relation_name'].str.replace(',|/', ' or', regex=True)
relation_df['relation_name'] = relation_df['relation_name'].str.replace('esp.','especially')
relation_df['relation_name'] = relation_df['relation_name'].str.replace('\(|\)|-|\.', '', regex=True)
relation_df['relation_label'] = relation_df['relation_name'].str.replace(' ', '_')

# Check if any relationshp names still have non alpha numeric values except space
error_relation_names = relation_df['relation_name'][relation_df['relation_name'].str.replace(' ', '').str.contains(r"[^a-zA-Z0-9]+", regex=True)].drop_duplicates()
if len(error_relation_names):
    print('Warning: The following relation names contain special characters, which can interfere with PyG/GraphSage')
    print(error_relation_names)
    
relation_df


# # 3) BioLinkBERT embedding

# In[8]:


from torch_geometric.data import HeteroData
from SIMP_LLM.llm_encode import EntityEncoder
from SIMP_LLM.dataloader_mappings import create_mapping, create_edges, embed_entities, embed_edges


# ### Set variables and load data

# In[9]:


drkg_entity_df


# In[10]:


## Set variables
run_full_sample = 1
Sample          = 5

if run_full_sample:
    # Run full DRKG
    entity_df = drkg_entity_df.copy()
    hrt_data = drkg_translated.copy()
    relation_lookup = relation_df.copy()
else:
    # Create relationship subset for testing
    test_relation_df = relation_df[relation_df['Connected entity-types'].isin(['Compound:Gene', 'Disease:Gene', 'Compound:Disease', 'Gene:Gene'])].copy()
    test_relation_df['relation_name'] = None

    activation_list = ['activation', 'agonism', 'agonism, activation'] 
    treat_list = ['Compound treats the disease', 'treats']
    gene_drug_list = ['inhibition']

    test_relation_df['relation_name'][test_relation_df['Interaction-type'].isin(activation_list)] = 'Compound activates gene'
    test_relation_df['relation_name'][test_relation_df['Interaction-type'].isin(treat_list)]      = 'Compound treats disease'
    test_relation_df['relation_name'][test_relation_df['Interaction-type'].isin(gene_drug_list)]   = 'Inhibition'
    test_relation_df = test_relation_df[~test_relation_df['relation_name'].isna()]
    print_head(test_relation_df)

    # Create test sample of DRKG relationships filtering to these relations (for full sample: delete and use drkg_entity_df)
    test_hrt_df = drkg_translated[drkg_translated[1].isin(test_relation_df['drkg_id'])]
    test_hrt_df = test_hrt_df.groupby(1).head(Sample).reset_index(drop=True)
    test_unique_entities = get_unique_entities(test_hrt_df, columns=[0,2])
    test_entity_df = drkg_entity_df[drkg_entity_df['name'].isin(test_unique_entities)]
    test_entity_df = test_entity_df[test_entity_df['entity_type'].isin(['Compound', 'Gene', 'Disease'])]
    print_head(test_hrt_df)
    print_head(test_entity_df)
    
    new_relation_df = relation_df.copy()
    new_relation_df['relation_name'] = new_relation_df['relation_name'].str.replace(' ', '_')


    entity_df = test_entity_df.copy()
    hrt_data = test_hrt_df.copy()
    relation_lookup = new_relation_df.copy()


# In[11]:


#new_relation_df[new_relation_df['relation_name'].str.contains('_')]


# In[12]:


hrt_data


# ### Build HeteroData Object

# In[13]:


import torch


# In[14]:


torch.cuda.is_available()


# In[15]:


# Initialize heterograph object
device   = "cuda"
Encoder  = EntityEncoder(device = device )
data = HeteroData()

# Embed entities, add to graph, and save embedding mapping dictionary of dictionaries
mapping_dict = embed_entities(entity_df, data, Encoder, device) 

# Embed relationships, add to graph, and save relation embeddings/mapping dictionary
relation_X, relation_mapping = embed_edges(hrt_data, relation_lookup, data, mapping_dict, Encoder, device)

# Print summary
#data = T.ToUndirected()(data)

#print(data)
for ent_type in entity_df['entity_type'].unique():
    print(f"Unique {ent_type}s: {len(mapping_dict[ent_type])} \t Matrix shape: {data[ent_type].x.shape }")
    # print(mapping_dict[ent_type]) # Prints whole dictionary so delete/uncomment if using all entities
    
del Encoder
data = T.ToUndirected()(data)
data.to("cpu")
print(data)

torch.save(mapping_dict, 'data2/mapping_dict')



