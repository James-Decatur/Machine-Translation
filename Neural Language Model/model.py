import torch.nn as nn

# Professor: Gongbo Tang
# Assignment 3 - Neural Language Models

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and an output layer."""

    def __init__(self, rnn_type, num_token, input_size, hidden_size, num_layers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.num_token = num_token
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(num_token, input_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU']""")
        self.outputLayer = nn.Linear(hidden_size, num_token)
        self.softmax = nn.LogSoftmax(dim=-1)

        if tie_weights:
            if hidden_size != input_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to emb_size')
            self.outputLayer.weight = self.embed.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)
        nn.init.zeros_(self.outputLayer.weight)
        nn.init.uniform_(self.outputLayer.weight, -initrange, initrange)

    def forward(self, input, hidden):
        
        # TODO: complement the forward computation,
        # given the input and the hidden states
        # return softmax results and the hidden states
        # hints: rnn -> dropout -> output layer -> log_softmax
        
        emb = self.drop(self.embed(input))
        output, hidden = self.rnn(emb,hidden)
        output = self.drop(output)
        output = self.outputLayer(output)
        output = output.view(-1, self.num_token)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                    weight.new_zeros(self.num_layers, bsz, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.hidden_size)


