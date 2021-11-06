class FNNModel(nn.Module):
    """
    make reference to the README for an explanation
    on what each layer does and why things are done 
    this way
    just remember that the equation is:
    y = (Wx + b) - this is one layer
    + (U tanh(d + Hx)) - this is the other layer
    """
    def __init__(self, ntoken, ninp, nhid, nbehind, batch_size, nlayers, emsize, dropout=0.3, tie_weights=False):
        super(FNNModel, self).__init__()
        self.model_type = 'FNN'
        self.batch_size = batch_size
        self.ntoken = ntoken
        self.twolayer = False
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # input layer
        self.flatten = nn.Flatten(start_dim=1) # concatenation 
        self.hidden_skip = nn.Linear(
            in_features=ninp*nbehind,
            out_features=ntoken
        ) # skip connection directly to output layer is just an FC
        self.hidden_non_linear = nn.Linear(
            in_features=ninp*nbehind,
            out_features=ntoken
        ) # similar to above but since we pass it through tanh, the output size is the size of the tanh layer
        selg.tanh = nn.Tanh()
        self.decoder = nn.Linear(in_features=nhid, out_features=ntoken, bias=False) # the simple matrix multiplication U
        self.init_weights()
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        return

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.hidden_skip.weight, -initrange, initrange)
        nn.init.uniform_(self.hidden_non_linear.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        input = torch.transpose(input, 1, 0)
        emb = self.encoder(input)
        emb = self.flatten(emb)
        # sorry i forgot what i called them above
        # im just going to use the equations xD
        Wb = self.drop(self.hidden_skip(emb))
        Hd =  self.drop(self.hidden_non_linear(emb))
        tanh = self.tanh(HD)
        U = self.decoder(tanh)
        decoded = Wb + U
        decoded = decoded.reshape(self.batch_size, self.ntoken)
        return F.log_softmax(decoded, dim=1)
