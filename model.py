import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super().__init__()
        
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size =embed_size, \
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bias=True,
                            batch_first=True,
                            dropout=0,
                            bidirectional=False
                           )
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    # Input:
    # features is of shape (batch_size, embed_size)
    # captions is of shape (num_word, num_captions)
    # Output:
    # outputs shape : (batch_size, caption length, vocab_size)
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]     
        
        # Initialize the hidden state
        self.batch_size = features.shape[0] 
        self.hidden = self.init_hidden(self.batch_size) 
                
        # Create embedded word vectors for each word in the captions
        # embeddings shape : (batch_size, captions length - 1, embed_size)
        embeddings = self.word_embeddings(captions) 
        
        # Stack the features and captions
        # new shape : (batch_size, caption length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs = []
        #features = inputs.unsqueeze(1)
        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            sampled_ids  = self.linear(hiddens.squeeze(1))       # outputs:  (batch_size, vocab_size)
            _, predicted = sampled_ids .max(1)                   # predicted: (batch_size)
            outputs.append(predicted.cpu().numpy()[0].item())
            
            if (predicted == 1):
                # We predicted the <end> word, so there is no further prediction to do
                break
                
            inputs = self.word_embeddings(predicted)             # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            
        #outputs = torch.stack(sampled_ids, max_len)              # sampled_ids: (batch_size, max_len)
        return outputs