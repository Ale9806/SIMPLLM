# AUTHORS: Alejandro

def forward_pass(model, link_predictor,data,return_node_emb:bool=False,prediction_entites:tuple=("compounds","disease")):
    ## If model is provided get GNN embeddings ##
    if model !=  None:                                        # If model is provided, embedd
        node_emb   = model(data.x_dict, data.edge_index_dict) # Embed Bert Embeddigns with graphsage (N, d) 
    else:                                                     # else 
        node_emb = x                                          #  use Bert default  Embedddings
   
    
    pred           = link_predictor(node_emb[prediction_entite[0]][edge_index[0]], node_emb[prediction_entite[1]][edge_index[0]])   
    if model !=  None and return_node_emb == True:
        return (pred,node_emb)
    else:
        return pred 


def train(model, link_predictor, data, optimizer,device:str="cuda"):
    """
    Runs offline training for model, link_predictor and node embeddings given the message
    edges and supervision edges.
    :param model: Torch Graph model used for updating node embeddings based on message passing 
        (If None, no embbeding is performed) 
    :param link_predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N, d) Initial node embeddings for all N nodes in graph
    :param edge_index: (2, E) Edge index for all edges in the graph

    :param optimizer: Torch Optimizer to update model parameters
    :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
    """
    if model != None: 
        model.train()
    link_predictor.train()
    train_losses = []

    optimizer.zero_grad()                                  # Reset Gradients
    #edge_index     = torch.tensor(edge_index).T           # Reshape edge index     (2,|E|)
    #x              = x.squeeze(dim=1)                     # Reshape Feature matrix (|N|,D)
    #x , edge_index = x.to(device) , edge_index.to(device) # Move data to devices


    ### Step 1: Get Embeddings:
    # Run message passing on the inital node embeddings to get updated embeddings

    ### This model has the option of only running link predictor without graphsage, for that case the node embedding
    ### is equal to the original embedding (X)
    pred           = forward_pass(model, link_predictor,data,return_node_emb=False)
    ground_truth   = data['compounds', 'treats', 'disease'].edge_label.to(device)

    
    
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth.unsqueeze(1))
    loss.backward()    # Backpropagate and update parameters
    optimizer.step()

    train_losses.append(loss.item())
    return sum(train_losses) / len(train_losses)