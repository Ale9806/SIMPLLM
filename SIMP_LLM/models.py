import torch         


import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric
import csv

from torch_geometric.nn import GCNConv, GATv2Conv

# class GCN(torch.nn.Module):
#   """Graph Convolutional Network"""
#   def __init__(self, dim_in, dim_h, dim_out):
#     super().__init__()
#     self.gcn1 = GCNConv(dim_in, dim_h)
#     self.gcn2 = GCNConv(dim_h, dim_out)
#     self.optimizer = torch.optim.Adam(self.parameters(),
#                                       lr=0.01,
#                                       weight_decay=5e-4)

#   def forward(self, x, edge_index):
#     h = F.dropout(x, p=0.5, training=self.training)
#     h = self.gcn1(h, edge_index)
#     h = torch.relu(h)
#     h = F.dropout(h, p=0.5, training=self.training)
#     h = self.gcn2(h, edge_index)
#     return h, F.log_softmax(h, dim=1)


# class GAT(torch.nn.Module):
#   """Graph Attention Network"""
#   def __init__(self, dim_in, dim_h, dim_out, heads=8):
#     super().__init__()
#     self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
#     self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
#     self.optimizer = torch.optim.Adam(self.parameters(),
#                                       lr=0.005,
#                                       weight_decay=5e-4)

#   def forward(self, x, edge_index):
#     h = F.dropout(x, p=0.6, training=self.training)
#     h = self.gat1(x, edge_index)
#     h = F.elu(h)
#     h = F.dropout(h, p=0.6, training=self.training)
#     h = self.gat2(h, edge_index)
#     return h, F.log_softmax(h, dim=1)




class GATStack(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, layers:int, dropout:float=0.3, return_embedding=True):
        """
            A stack of GraphSAGE Module 
            input_dim        <int>:   Input dimension
            hidden_dim       <int>:   Hidden dimension
            output_dim       <int>:   Output dimension
            layers           <int>:   Number of layers
            dropout          <float>: Dropout rate
            return_embedding <bool>:  Whether to return the return_embeddingedding of the input graph
        """
        
        super(GATStack, self).__init__()
        graphSage_conv               =  GATv2Conv
        self.dropout                 = dropout
        self.layers                  = layers
        self.return_embedding        = return_embedding
        #self.training                = train

        ### Initalize the layers ###
        self.convs                   = nn.ModuleList()                      # ModuleList to hold the layers
        for l in range(self.layers):
            if l == 0:
                ### First layer  maps from input_dim to hidden_dim ###
                self.convs.append(graphSage_conv(input_dim, hidden_dim,add_self_loops= False ))
            else:
                ### All other layers map from hidden_dim to hidden_dim ###
                self.convs.append(graphSage_conv(hidden_dim, hidden_dim,add_self_loops= False))

        # post-message-passing processing MLP
        self.post_mp = nn.Sequential(
                                     nn.Linear(hidden_dim, hidden_dim), 
                                     nn.Dropout(self.dropout ),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return final layer of return_embeddingeddings if specified
        if self.return_embedding:
            return x

        # Else return class probabilities
        return F.log_softmax(x, dim=1)

    #def loss(self, pred, label):
    
    

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, layers:int, dropout:float=0.3, return_embedding=True):
        """
            A stack of GraphSAGE Module 
            input_dim        <int>:   Input dimension
            hidden_dim       <int>:   Hidden dimension
            output_dim       <int>:   Output dimension
            layers           <int>:   Number of layers
            dropout          <float>: Dropout rate
            return_embedding <bool>:  Whether to return the return_embeddingedding of the input graph
        """
        
        super(GNNStack, self).__init__()
        graphSage_conv               = pyg.nn.SAGEConv
        self.dropout                 = dropout
        self.layers                  = layers
        self.return_embedding        = return_embedding
        #self.training                = train

        ### Initalize the layers ###
        self.convs                   = nn.ModuleList()                      # ModuleList to hold the layers
        for l in range(self.layers):
            if l == 0:
                ### First layer  maps from input_dim to hidden_dim ###
                self.convs.append(graphSage_conv(input_dim, hidden_dim))
            else:
                ### All other layers map from hidden_dim to hidden_dim ###
                self.convs.append(graphSage_conv(hidden_dim, hidden_dim))

        # post-message-passing processing MLP
        self.post_mp = nn.Sequential(
                                     nn.Linear(hidden_dim, hidden_dim), 
                                     nn.Dropout(self.dropout ),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return final layer of return_embeddingeddings if specified
        if self.return_embedding:
            return x

        # Else return class probabilities
        return F.log_softmax(x, dim=1)

    #def loss(self, pred, label):
    #    return F.nll_loss(pred, label)
    


class LinkPredictorMLP(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, n_layers:int,dropout_probabilty:float=0.3):
        """
        Args:
            in_channels (int):     Number of input features.
            hidden_channels (int): Number of hidden features.
            out_channels (int):    Number of output features.
            n_layers (int):        Number of MLP layers.
            dropout (float):       Dropout probability.
            """
        super(LinkPredictorMLP, self).__init__()
        self.dropout_probabilty    = dropout_probabilty  # dropout probability
        self.mlp_layers            = nn.ModuleList()     # ModuleList: is a list of modules
        self.non_linearity         = F.relu              # non-linearity
        
        for i in range(n_layers - 1):                                 
            if i == 0:
                self.mlp_layers.append(nn.Linear(in_channels, hidden_channels))          # input layer (in_channels, hidden_channels)
            else:
                self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))      # hidden layers (hidden_channels, hidden_channels)

        self.mlp_layers.append(nn.Linear(hidden_channels, out_channels))                 # output layer (hidden_channels, out_channels)


    def reset_parameters(self):
        for mlp_layer in self.mlp_layers:
            mlp_layer.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j                                                     # element-wise multiplication
        for mlp_layer in self.mlp_layers[:-1]:                            # iterate over all layers except the last one
            x = mlp_layer(x)                                              # apply linear transformation
            x = self.non_linearity(x)                                     # Apply non linear activation function
            x = F.dropout(x, p=self.dropout_probabilty,training=self.training)      # Apply dropout
            #x = F.dropout(x, p=self.dropout_probabilty)      # Apply dropout
        x = self.mlp_layers[-1](x)                                        # apply linear transformation to the last layer
        x = torch.sigmoid(x)                                              # apply sigmoid activation function to get the probability
        return x
    
    
class CosineSimilarityModel(nn.Module):
    def __init__(self, input_dim):
        super(CosineSimilarityModel, self).__init__()
        #self.fc  = nn.Linear(input_dim, 1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x1, x2):
        # Normalize the input vectors
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        #print(x1.shape)
        
        # Compute the cosine similarity
        similarity = self.cos(x1, x2)
       # print(x1.shape)
        
        # Pass through a linear layer
        #output = self.fc(similarity)
        
        # Apply sigmoid activation to get the final similarity prediction
        prediction = torch.sigmoid(similarity)
        
        
        return prediction


### We will use This function to save our best model during trainnig ###
def save_torch_model(model,epoch,PATH:str,optimizer):
    print(f"Saving Model in Path {PATH}")
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer':optimizer,      
                }, PATH)