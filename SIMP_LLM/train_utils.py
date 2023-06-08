
# AUTHORS: Alejandro

import torch
import matplotlib.pyplot as plt #needed to visualize loss curves
import numpy as np 

from sklearn         import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc 

from   torch_geometric.utils import negative_sampling
import torch.nn.functional as F


#########  MASKING  NODE  FUNCTIONS ##############
def filter_edge_index_by_value(edge_index, value,return_mask=True):
    mask = edge_index[0] == value
    filtered_edge_index = edge_index[:, mask]
    if return_mask:
        return filtered_edge_index, mask
    else:
        return filtered_edge_index


def split_edge_index(edge_index ,percentage:float   = 0.9,verbose:bool=False):
    edge_index_1 = torch.tensor([],dtype=torch.long)    # Define First Edge Index
    edge_index_2 = torch.tensor([],dtype=torch.long)    # Define Second Edge Index
    
    unique_drugs= torch.unique(edge_index[0])                             # Get unique drugs and d = iseases
    for d in unique_drugs:                                                # Iterate over all unique drugs
        fiter_edge_idex,mask = filter_edge_index_by_value(edge_index,d)   # Get drugs repeated
        number_of_drugs      = fiter_edge_idex.shape[1]                   # Get number of Drugs
    
        limit       = int(number_of_drugs *percentage)                                # get split:  [0:limit] - [limit:]
     
        edge_index_1 =   torch.cat((edge_index_1, fiter_edge_idex[:,0:limit]),dim=1)  # Set edge_index_1
        edge_index_2 =   torch.cat((edge_index_2, fiter_edge_idex[:,limit:]),dim=1)   # Set edge_index_2

    if verbose:
        # Print the results
        print("##############")
        print(f"Edge Index 1: {edge_index_1.shape}")
       # print(edge_index_1)

        print(f"\nEdge Index 2: {edge_index_2.shape}")
        #print(edge_index_2)
        print("##############")

    assert edge_index_1.shape[1] + edge_index_2.shape[1] == edge_index.shape[1]
    
    return  edge_index_1 , edge_index_2


def get_negative_edges(data,number_of_samples,triplet=("Compound", "Compound_treats_the_disease", "Disease")):
    edge_index = data[triplet].edge_index 
    x          =  data[triplet[0]].x
    y          =  data[triplet[2]].x
    
    neg_edge = negative_sampling(edge_index       = edge_index,               # Possitve PPI's
                                 num_nodes        = (x.shape[0],y.shape[0]),  # Total number of nodes in graph
                                 num_neg_samples  = number_of_samples,        # Same Number of edges as in positive example
                                 method           = 'dense',                  # Method for edge generation
                                 force_undirected = True)                     # Our graph is undirected
    return neg_edge
#########################################################################




########## Train Loops ####

def forward_pass(model, link_predictor,data_embed,data_predict,data_sample,return_node_emb:bool=False,prediction_entites:tuple=("Compound","Disease"),device="cpu"):
    ## If model is provided get GNN embeddings ##
    data_embed = data_embed.to(device)
    if model !=  None:                                                     # If model is provided, embedd
        node_emb   = model(data_embed.x_dict, data_embed.edge_index_dict)  # Embed Bert Embeddigns with graphsage (N, d) 
    else:                                                                  # else 
        node_emb = x                                                       #  use Bert default  Embedddings
   
    # Positive Edge _index
    pos_edge_index = data_predict['Compound', 'Compound_treats_the_disease', 'Disease'].edge_index
    
    
    # Negative Edge index
    negative_edge_index = get_negative_edges(data_sample,pos_edge_index.shape[1])
    
    
    
    edge_index          = torch.cat((pos_edge_index, negative_edge_index), dim=1)
    head                = node_emb[prediction_entites[0]][edge_index[0]].to(device)
    tail                = node_emb[prediction_entites[1]][edge_index[1]].to(device)
    pred                = link_predictor( head  ,    tail   )   
    
    ones_tensor         = torch.ones(pos_edge_index.shape[1])
    zero_tensor         = torch.zeros(negative_edge_index.shape[1])
    labels              = torch.cat((ones_tensor ,zero_tensor), dim=0)
    
    
    
    if model !=  None and return_node_emb == True:
        return (pred,labels.to(device),head,tail,negative_edge_index )
    else:
        return pred,labels.to(device)


    

def train(model, link_predictor, data_embed,data_predict,data_sample, optimizer,triplet:tuple=('Compound', 'Compound_treats_the_disease', 'Disease'),device:str="cuda",head="COSINE"):
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
    pred,ground_truth   = forward_pass(model, link_predictor,data_embed= data_embed,data_predict=data_predict,data_sample=data_sample,return_node_emb=False)
    ground_truth       =  ground_truth.to(device)
 
    if head == "MLP" or  head ==  "MLP_ONLY"  :
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth.unsqueeze(1).float())
    if head == "COSINE":
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth.float())
    loss.backward()    # Backpropagate and update parameters
    optimizer.step()

    train_losses.append(loss.item())
    return sum(train_losses) / len(train_losses)



import pdb
def train_with_triplet(model, link_predictor, data_embed,data_predict,data_sample, optimizer,triplet_loss,triplet:tuple=('Compound', 'Compound_treats_the_disease', 'Disease'),device:str="cuda",lambda_=0.7):
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
    pred,ground_truth,head,tail,negative_edge_index    = forward_pass(model, link_predictor,data_embed= data_embed,data_predict=data_predict,data_sample=data_sample,return_node_emb=True)
    ground_truth                                        =  ground_truth.to(device)
    
    
    # head halg is positive other half is negative
    HEADS          = head[0:head.shape[0]//2,:]
    Positive_links = tail[0:head.shape[0]//2,:]
    Negative_links = tail[head.shape[0]//2:,:]
    
    perm             = torch.randperm(len(Negative_links))
    Negative_links = Negative_links[perm]

    #pdb.set_trace()

  

    loss_a = F.binary_cross_entropy_with_logits(pred, ground_truth.float())
    loss_b = triplet_loss(  HEADS , Positive_links, Negative_links )
    loss   = lambda_*loss_a + (1 -lambda_)*loss_b 
    #loss   =  (1 -lambda_)*loss_b 
    loss.backward()    # Backpropagate and update parameters
    optimizer.step()
    
    

    train_losses.append(loss.item())
    return sum(train_losses) / len(train_losses)




def evaluate(model,predictor,data_val_embed,data_val_predict,data_sample,threshold:float=0.5, show_extra_metrics=True,return_dict=False):
    model.eval()
    predictor.eval()
    with torch.no_grad():
        pred,ground_truth = forward_pass(model, predictor,data_val_embed,data_val_predict,data_sample=data_sample)
        acc        = accuracy_score(pred.to("cpu")  > threshold  ,ground_truth.to("cpu") )
        
    if show_extra_metrics == True:
        fig, ax = plt.subplots(1, 2,figsize=(10,2))
        fpr, tpr, thresholds = metrics.roc_curve( ground_truth, pred)
        
        auc_  =  auc(fpr, tpr)

        
        sens      =  tpr
        spec      =  1 - fpr
        j         = sens + spec -1
        opt_index = np.where(j == np.max(j))[0][0]
        op_point  = thresholds[opt_index]
        
        print(f"Youdens  index: {op_point:.4f} Sensitivity: {round(sens[opt_index],4)} Specificity: {round(spec[opt_index],4) } AUC: {auc_}")
       
        ax[0].set_title("ROC Curve")
        #ax[1].set_title("Confusion Matrix")
        if model == None:
            ax[0].plot(fpr,tpr,label="MLP") 
        else:
            ax[0].plot(fpr,tpr,label="GraphSage+COS") 
        ax[0].plot([0, 1], [0, 1], 'k--')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].legend()
       
    
        cfm = metrics.confusion_matrix(ground_truth, np.array(pred)> op_point)
        
        cmn = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis] # Normalise
        disp = ConfusionMatrixDisplay(cmn)
        disp.plot(ax=ax[1])
        
        
        
        plt.show()
        
    if return_dict:
        return  {"FPR": fpr.tolist(),"TPR":tpr.tolist(),"Sensitiviy":sens.tolist(),"Specificity":spec.tolist() ,"j": j.tolist(),"opitimal_point":(opt_index,op_point),"CFM": (cfm,cmn),"Acc":acc}
    else:
        return acc