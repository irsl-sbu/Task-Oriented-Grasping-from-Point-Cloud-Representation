# Class for specifying the desired Neural Network Architecture.
# By: Aditya Patankar 

from torch import nn
import torch
import math
import numpy as np
import torch.nn.functional as F

# This is not the exact neural network architecture for our metricNN. This is just the trial version:
class metric_nn(nn.Module):

    # In the __init__ function we define all the layers that we want to have:
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(metric_nn, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.GELU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.GELU()
        self.linear3 = nn.Linear(hidden_size2, 1)


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        y_pred = self.linear3(out)
        return y_pred

'''The above method is one way of defining the neural network. 
   The second method is to use the activation functions directly in the forward pass function to the output of the linear layers 
   using the torch API.'''

class MLP(nn.Module): # how many stacks we can generate

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        

class metric_nn_generic(nn.Module):

    # In the __init__ function we define all the layers that we want to have:
    def __init__(self, input_size, hidden_size=8, depth=2, norm=nn.LayerNorm, act_layer=nn.GELU, residual=False, post_norm=False, drop=0.1):
        '''
            Creates an MLP Network Of Number of Linear Layers == depth*2
            input_size: int, size of input feature vector
            hidden_size: int, size of hidden layer vector (will be multipled by 2)
            depth: int, how many linear layers wanted to be added where total_layers = depth*2
            norm: torch.nn class, the type of normalization, either LayerNorm, Batchnorm1d or None
            act_layer: torch.nn class, activation function for non-linearity: nn.GELU, nn.ReLU, nn.Tanh, nn.SiLU     
            residual: boolean, whether to use skip connections addition in inference, only avaiable when depth >= 3
        '''
    
        
        super(metric_nn_generic, self).__init__()        
        
        # Depth is the number of intermediate layers the model will have
        assert depth >= 1 # must be greater than or equal to 2
    
        self.depth = depth
        self.residual = residual
        self.post_norm = post_norm
        
        if depth >= 2:
            self.embedding_mlp = MLP(in_features=input_size, hidden_features=hidden_size, out_features=hidden_size*2, act_layer=act_layer)
            
            start_size=hidden_size*2
           
            self.hidden_mlps = []            
            
            for i in range(depth-2):      
                         
                hidden_mlp = MLP(in_features=start_size, hidden_features=hidden_size*(i+2), out_features=start_size, act_layer=act_layer, drop=drop)
                
                self.hidden_mlps.append(hidden_mlp)
                if norm != None:
                    self.hidden_mlps.append(norm(hidden_mlp.out_features))
                else: 
                    self.hidden_mlps.append(nn.Identity()) # when it is None, normalization is identity
                
            
            self.hidden_mlps = nn.Sequential(*self.hidden_mlps)

            if self.depth > 2:
                self.regressor_mlp = MLP(self.hidden_mlps[-2].out_features, hidden_features=start_size*2, out_features=1, act_layer=act_layer)
            else:                 
                self.regressor_mlp = MLP(self.embedding_mlp.out_features, hidden_features=start_size*2, out_features=1, act_layer=act_layer)
        else:
            self.regressor_mlp = MLP(input_size, hidden_features=hidden_size*2, out_features=1, act_layer=act_layer)
            
        
    def forward(self, x):        
               
        if self.depth > 2: 
            
            embedding = self.embedding_mlp(x)
            
            if self.residual:
                for i in range(len(self.hidden_mlps)): 
                    if i % 2 == 0:
                        if self.post_norm:
                            embedding = embedding + self.hidden_mlps[i+1](self.hidden_mlps[i](embedding))
                        else: 
                            embedding = embedding + self.hidden_mlps[i](self.hidden_mlps[i+1](embedding)) # post-norm, pass through mlp ---> norm
            else: 
                
                embedding = self.hidden_mlps(embedding)
            
            
            y_pred = self.regressor_mlp(embedding)      
        elif self.depth == 2: 
            embedding = self.embedding_mlp(x)
            y_pred = self.regressor_mlp(embedding)   
        else: 
            y_pred = self.regressor_mlp(x)   
        
        return y_pred

        
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class multi_head_attention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        
        
        batch_size, seq_length = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        breakpoint()
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class encoder_block(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = multi_head_attention(input_dim, input_dim, num_heads)

        # Two-layer MLP - replace with my own MLP implementation?
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class transformer_encoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([encoder_block(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class positional_encoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
       
        x = x + self.pe[:, :x.size(1)]
        return x

class transformer_model(nn.Module):

    def __init__(self, projection_mlp, positional_encoding, transformer, regressor, add_positional_encoding=True):
        super().__init__()
        self.projection_mlp = projection_mlp 
        self.positional_encoding = positional_encoding
        self.transformer = transformer 
        self.regressor = regressor
        self.add_positional_encoding = add_positional_encoding

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.projection_mlp(x)
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.regressor(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.projection_mlp(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps