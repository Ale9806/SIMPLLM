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


# ## GRAPH SAGE

# In[16]:


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric as pyg
# import torch_geometric
# from torch_geometric.nn import SAGEConv, to_hetero
# from   torch.utils.data      import Dataset, DataLoader
# from   torch_geometric.data  import Data
# from   torch_geometric.utils import negative_sampling

# from torch_geometric.nn import SAGEConv, to_hetero




# class GNNStack(torch.nn.Module):
#     def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, layers:int, dropout:float=0.3, return_embedding=False):
#         """
#             A stack of GraphSAGE Module 
#             input_dim        <int>:   Input dimension
#             hidden_dim       <int>:   Hidden dimension
#             output_dim       <int>:   Output dimension
#             layers           <int>:   Number of layers
#             dropout          <float>: Dropout rate
#             return_embedding <bool>:  Whether to return the return_embeddingedding of the input graph
#         """
        
#         super(GNNStack, self).__init__()
#         graphSage_conv               = pyg.nn.SAGEConv
#         self.dropout                 = dropout
#         self.layers                  = layers
#         self.return_embedding        = return_embedding

#         ### Initalize the layers ###
#         self.convs                   = nn.ModuleList()                      # ModuleList to hold the layers
#         for l in range(self.layers):
#             if l == 0:
#                 ### First layer  maps from input_dim to hidden_dim ###
#                 self.convs.append(graphSage_conv(input_dim, hidden_dim))
#             else:
#                 ### All other layers map from hidden_dim to hidden_dim ###
#                 self.convs.append(graphSage_conv(hidden_dim, hidden_dim))

#         # post-message-passing processing MLP
#         self.post_mp = nn.Sequential(
#                                      nn.Linear(hidden_dim, hidden_dim), 
#                                      nn.Dropout(self.dropout),
#                                      nn.Linear(hidden_dim, output_dim))

#     def forward(self, x, edge_index):
#         for i in range(self.layers):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.post_mp(x)

#         # Return final layer of return_embeddingeddings if specified
#         if self.return_embedding:
#             return x

#         # Else return class probabilities
#         return F.log_softmax(x, dim=1)

#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)
    


# class LinkPredictorMLP(nn.Module):
#     def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, n_layers:int,dropout_probabilty:float=0.3):
#         """
#         Args:
#             in_channels (int):     Number of input features.
#             hidden_channels (int): Number of hidden features.
#             out_channels (int):    Number of output features.
#             n_layers (int):        Number of MLP layers.
#             dropout (float):       Dropout probability.
#             """
#         super(LinkPredictorMLP, self).__init__()
#         self.dropout_probabilty    = dropout_probabilty  # dropout probability
#         self.mlp_layers            = nn.ModuleList()     # ModuleList: is a list of modules
#         self.non_linearity         = F.relu              # non-linearity
        
#         for i in range(n_layers - 1):                                 
#             if i == 0:
#                 self.mlp_layers.append(nn.Linear(in_channels, hidden_channels))          # input layer (in_channels, hidden_channels)
#             else:
#                 self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))      # hidden layers (hidden_channels, hidden_channels)

#         self.mlp_layers.append(nn.Linear(hidden_channels, out_channels))                 # output layer (hidden_channels, out_channels)


#     def reset_parameters(self):
#         for mlp_layer in self.mlp_layers:
#             mlp_layer.reset_parameters()

#     def forward(self, x_i, x_j):
#         x = x_i * x_j                                                     # element-wise multiplication
#         for mlp_layer in self.mlp_layers[:-1]:                            # iterate over all layers except the last one
#             x = mlp_layer(x)                                              # apply linear transformation
#             x = self.non_linearity(x)                                     # Apply non linear activation function
#             x = F.dropout(x, p=self.dropout_probabilty,training=self.training)      # Apply dropout
#         x = self.mlp_layers[-1](x)                                        # apply linear transformation to the last layer
#         x = torch.sigmoid(x)                                              # apply sigmoid activation function to get the probability
#         return x
    
# ### We will use This function to save our best model during trainnig ###
# def save_torch_model(model,epoch,PATH:str,optimizer):
#     print(f"Saving Model in Path {PATH}")
#     torch.save({'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer':optimizer,      
#                 }, PATH)


# # In[17]:


# epochs        = 500
# hidden_dim    = 524      # 256 
# dropout       = 0.7
# num_layers    = 3
# learning_rate = 1e-4
# node_emb_dim  = 768



# HomoGNN         = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embedding=True).to(device) # the graph neural network that takes all the node embeddings as inputs to message pass and agregate
# HeteroGNN       = to_hetero(HomoGNN   , data.metadata(), aggr='sum')
# link_predictor  = LinkPredictorMLP(hidden_dim, hidden_dim, 1, num_layers , dropout).to(device) # the MLP that takes embeddings of a pair of nodes and predicts the existence of an edge between them
# #optimizer      = torch.optim.AdamW(list(model.parameters()) + list(link_predictor.parameters() ), lr=learning_rate, weight_decay=1e-4)
# optimizer       = torch.optim.Adam(list(HeteroGNN.parameters()) + list(link_predictor.parameters() ), lr=learning_rate)

# print(HeteroGNN )
# print(link_predictor)
# print(f"Models Loaded to {device}")
# data.to("cuda")
# HeteroGNN.to("cuda")


# # In[18]:


# node_emb   = HeteroGNN(data.x_dict, data.edge_index_dict)
# edge_index = data['Compound', 'Compound_treats_the_disease', 'Disease'].edge_index 
# pos_pred   = link_predictor(node_emb["Compound"][edge_index[0]], node_emb["Disease"][edge_index[1]])   # (B, )



# # In[19]:


# def forward_pass(model, link_predictor,data,return_node_emb:bool=False,prediction_entites:tuple=("Compounds","Disease")):
#     ## If model is provided get GNN embeddings ##
#     if model !=  None:                                        # If model is provided, embedd
#         node_emb   = model(data.x_dict, data.edge_index_dict) # Embed Bert Embeddigns with graphsage (N, d) 
#     else:                                                     # else 
#         node_emb = x                                          #  use Bert default  Embedddings
   
    
#     pred           = link_predictor(node_emb[prediction_entite[0]][edge_index[0]], node_emb[prediction_entite[1]][edge_index[0]])   
#     if model !=  None and return_node_emb == True:
#         return (pred,node_emb)
#     else:
#         return pred 


# def train(model, link_predictor, data, optimizer,device:str="cuda"):
#     """
#     Runs offline training for model, link_predictor and node embeddings given the message
#     edges and supervision edges.
#     :param model: Torch Graph model used for updating node embeddings based on message passing 
#         (If None, no embbeding is performed) 
#     :param link_predictor: Torch model used for predicting whether edge exists or not
#     :param emb: (N, d) Initial node embeddings for all N nodes in graph
#     :param edge_index: (2, E) Edge index for all edges in the graph

#     :param optimizer: Torch Optimizer to update model parameters
#     :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
#     """
#     if model != None: 
#         model.train()
#     link_predictor.train()
#     train_losses = []

#     optimizer.zero_grad()                                  # Reset Gradients
#     #edge_index     = torch.tensor(edge_index).T           # Reshape edge index     (2,|E|)
#     #x              = x.squeeze(dim=1)                     # Reshape Feature matrix (|N|,D)
#     #x , edge_index = x.to(device) , edge_index.to(device) # Move data to devices


#     ### Step 1: Get Embeddings:
#     # Run message passing on the inital node embeddings to get updated embeddings

#     ### This model has the option of only running link predictor without graphsage, for that case the node embedding
#     ### is equal to the original embedding (X)
#     pred           = forward_pass(model, link_predictor,data,return_node_emb=False)
#     ground_truth   = data['compounds', 'treats', 'disease'].edge_label.to(device)

    
    
#     loss = F.binary_cross_entropy_with_logits(pred, ground_truth.unsqueeze(1))
#     loss.backward()    # Backpropagate and update parameters
#     optimizer.step()

#     train_losses.append(loss.item())
#     return sum(train_losses) / len(train_losses)


# # In[22]:


# data.to("cpu")


# # In[23]:


# data['Compound', 'Compound_treats_the_disease', 'Disease'].edge_label = torch.ones(data['Compound', 'Compound_treats_the_disease', 'Disease'].edge_index.shape[1], dtype=torch.long)
# transform = T.RandomLinkSplit(
#     num_val=0.1,
#     num_test=0.1,

   
#     edge_types=("Compound", "Compound_treats_the_disease", "Disease"),
   
   
# )

# train_data, val_data, test_data = transform(data)


# print(f"Train Data:\n{train_data}")
# print(f"Validation Data:\n{val_data}")
# print(f"Test Data:\n{test_data}")


# # In[110]:





# # In[ ]:




