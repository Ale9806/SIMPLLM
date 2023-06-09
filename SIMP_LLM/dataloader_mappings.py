### AUTHORS: Alejandro, Selina, and Rohan; authorship listed by function

import torch 
import pandas as pd 
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import csv
import torch.nn as nn

# AUTHORS: Alejandro
def create_mapping(entity_list: list, encoder=None, batch_size=64, device=None) -> dict:
    """
    Arguments:
        entity_list <list>: a list of all entity elements (e.g. all drugs in dataset)
        encoder <callable>: function to encode each entity element using a BERT-like model (default: None)
        batch_size <int>: the batch size for encoding (default: 64)
        device <str>: the device to use for encoding (default: None, i.e., use CPU)
    Output:
        mapping <dict>: a mapping from each entity element to its index in the list
        encoded_entities <torch.Tensor>: a tensor of shape (len(entity_list), encoder_output_size)
                                          containing the encoded representations of the entities.
                                          If encoder is None, this will be None as well.
    """

    entity_list = list(set(entity_list))  # Convert to set to remove duplicates, then back to list
    mapping = {index: i for i, index in enumerate(entity_list)}


    if encoder is not None:
        num_entities = len(entity_list)
        num_batches = (num_entities + batch_size - 1) // batch_size

        encoded_entities = []
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_entities)
            batch_entities = entity_list[batch_start:batch_end]

            if device is not None:
                batch_entities = batch_entities
            
            batch_encodings = encoder(batch_entities)

            if device is not None:
                batch_encodings = batch_encodings.to(device)

            encoded_entities.append(batch_encodings)

        encoded_entities = torch.cat(encoded_entities, dim=0)

    else:
        encoded_entities = None
        
    

    return  encoded_entities.to("cpu"), mapping


# AUTHORS: Selina and Alejandro - although not used
def embed_nodes(df, encoders=None, **kwargs):
    '''
    Embeds values of dataframe and creates mapping using specified encoder.
    Equivalent to load_node_csv() here: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
    THIS MAY BE OBSOLETE
    '''
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping



# AUTHORS: Alejandro
def create_edges(df, src_index_col, src_mapping, dst_index_col, dst_mapping,edge_attr=None):
    '''
    Creates index matrix and edge attribute
    '''
    src        = [src_mapping[index] for index in df[src_index_col]]
    dst        = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)


    if edge_attr is not None:
      edge_attr = edge_attr.repeat(len(edge_index[0]), 1)

    return edge_index,edge_attr
        

# AUTHORS: Selina, with one line torch.save edit by Rohan and Alejandro
def embed_entities(entity_df, graph_obj, encoder, device):
    '''Embeds entities, inputs embeddings directly into Heterograph object, and returns mapping dictionary (which is a dictionary of dictionaries) by entity type'''
    
    entity_lookup = entity_df.copy()
    mapping_dict = {}

    for entity in entity_lookup['entity_type'].unique():                                        # For each entity type
        entity_names = entity_lookup.loc[entity_lookup['entity_type'] == entity, 'name']        # Get entity names associated with entity type
        entity_X, entity_mapping = create_mapping(entity_names, encoder=encoder, device=device) # Maps entities to indices
        graph_obj[entity].x = entity_X                                                          # Assign entity type embeddings to graph object
        torch.save(entity_X, f"data2/ckpts/entity/{entity}.pt")
        
        mapping_dict[entity] = entity_mapping                                                   # Add entity type mapping to overall mapping dictionary
    
    return mapping_dict


# AUTHORS: Selina primary, with file save (torch.save) and save to CPU edits by Rohan and Alejandro
def embed_edges(hrt_data, relation_lookup, graph_obj, mapping_dict, encoder, device):
    '''
    Given dataframe with columns for head-relationship-tail (h,r,t) in that order, create edges in Heterograph object by relationship type.
    Assumes entity types are already embedded in graph.
    Requires relation lookup table mapping DRKG relation name to natural language relation name, in addition to head and tail entity types for each relation.
    Returns relation tensor and mapping while directly embedding graph.
    '''
    # Create mapping for relations
    relation_lookup_subset       = relation_lookup[relation_lookup['drkg_id'].isin(hrt_data[1])]  # Only use relations that are in the hrt data
    relation_name_list           = relation_lookup_subset['relation_name'].unique()  # Only use relations that are in the hrt data
    relation_X, relation_mapping = create_mapping(relation_name_list,encoder=encoder,device=device)

    torch.save(relation_X, 'data2/relation_X')
    torch.save(relation_mapping, 'data2/relation_mapping')

    # By entity-entity pair
    for ent_types in relation_lookup_subset['Connected entity-types'].unique():
        relation_lookup_sub_subset = relation_lookup_subset[relation_lookup_subset['Connected entity-types'] == ent_types]

        # By relation name within entity-entity pair (since relation names can be common across different pairs)
        for relation_name in relation_lookup_sub_subset['relation_name'].unique():
            # Get relation codes associated with relation type and filter knowledge graph to associated relation codes

            relation_subset = relation_lookup_sub_subset[relation_lookup_sub_subset['relation_name'] == relation_name]  
            hrt_subset = hrt_data[hrt_data[1].isin(relation_subset['drkg_id'])]   

            # If no entries in translated DRKG for relation name, continue
            if len(hrt_subset) == 0:
                continue

            # Get head and tail entity types from data as well as relation_label
            head_entity = relation_subset['head_entity'].unique()
            tail_entity = relation_subset['tail_entity'].unique()
            relation_label = relation_subset['relation_label'].unique()

            if len(head_entity) > 1 or len(tail_entity) > 1:
                raise Exception(f"Multiple types of entities for the following relationship: {relation_name}")
            
            if len(relation_label) > 1:
                raise Exception(f"Multiple relation labels for the following relationship: {relation_name}")
            
            head_entity = head_entity[0]
            tail_entity = tail_entity[0]
            relation_label = relation_label[0]
            
            # Create edge attributes for graph
            relation_feature = relation_X[relation_mapping[relation_name],:].reshape(1,-1)
            Edge_index,edge_attribute = create_edges(df            = hrt_subset,
                                                    src_index_col  = 0, 
                                                    src_mapping    = mapping_dict[head_entity] , 
                                                    dst_index_col  = 2, 
                                                    dst_mapping    = mapping_dict[tail_entity] ,
                                                    edge_attr      = relation_feature)

            graph_obj[head_entity, relation_label, tail_entity].edge_index = Edge_index
            graph_obj[head_entity, relation_label, tail_entity].edge_label = edge_attribute
            torch.save(Edge_index, f"data2/ckpts/edge_index/{head_entity}_{relation_label}_{tail_entity}.pt")
            torch.save(edge_attribute, f"data2/ckpts/edge_attribute/{head_entity}_{relation_label}_{tail_entity}.pt")
            

        graph_obj.to('cpu')
        torch.save(graph_obj, 'data2/sage/graph_obj')

    return relation_X, relation_mapping


# AUTHORS: Rohan, with correction by Selina and Alejandro for undirected graph
import pdb
def load_graph(triplets,path="data2",random_embeddings:bool = False):
    """
    Triplets must be (h, r, t)
    """
    graph_obj = HeteroData()
    print(path)
    for (h, r, t) in triplets:
        
        if random_embeddings:
            head    =  torch.load(f"{path}/ckpts/entity/{h}.pt")
            n_m     = head.shape
            head_   = torch.empty(n_m[0], n_m[1])
            nn.init.uniform_(head_)
            graph_obj[h].x = head_ 
            
            tail    = torch.load(f"{path}/ckpts/entity/{t}.pt")
            n_m     = tail.shape
            tail_   = torch.empty(n_m[0], n_m[1])
            nn.init.uniform_(tail_)
            graph_obj[t].x = tail_
           
        else:
            graph_obj[h].x = torch.load(f"{path}/ckpts/entity/{h}.pt")
            graph_obj[t].x = torch.load(f"{path}/ckpts/entity/{t}.pt")
        
        
        graph_obj[h, r, t].edge_index =  torch.load(f"data2/ckpts/edge_index/{h}_{r}_{t}.pt")
    
    #pdb.set_trace()
        


    graph_obj = T.ToUndirected()(graph_obj)

    return graph_obj

    

# AUTHORS: Alejandro
def load_csv_as_list(file_path):
    data = []
    set_ = set()
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if row[3] == 'Atc' or row[3] == "Tax":
                continue
            else:
                data.append((row[1],row[2],row[3]))
    return data