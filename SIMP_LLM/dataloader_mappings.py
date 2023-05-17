import torch 
import pandas as pd 
import numpy as np

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
                batch_encodings = batch_encodings.to('cpu')

            encoded_entities.append(batch_encodings)

        encoded_entities = torch.cat(encoded_entities, dim=0)

    else:
        encoded_entities = None

    return  encoded_entities, mapping

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



def create_edges(df, src_index_col, src_mapping, dst_index_col, dst_mapping,edge_attr=None):
    '''
    THIS MAY BE OBSOLETE
    '''
    src        = [src_mapping[index] for index in df[src_index_col]]
    dst        = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    if edge_attr is not None:
      edge_attr = edge_attr.repeat(len(edge_index[0]), 1)

    return edge_index,edge_attr
        

def embed_entities(entity_df, graph_obj, Encoder, device):
    '''Embeds entities, inputs embeddings directly into Heterograph object, and returns mapping dictionary (which is a dictionary of dictionaries) by entity type'''
    
    entity_lookup = entity_df.copy()
    mapping_dict = {}

    for entity in entity_lookup['entity_type'].unique():                                        # For each entity type
        entity_names = entity_lookup.loc[entity_lookup['entity_type'] == entity, 'name']        # Get entity names associated with entity type
        entity_X, entity_mapping = create_mapping(entity_names, encoder=Encoder, device=device) # Maps entities to indices
        graph_obj[entity].x = entity_X                                                          # Assign entity type embeddings to graph object
        mapping_dict[entity] = entity_mapping                                                   # Add entity type mapping to overall mapping dictionary
    
    return mapping_dict

