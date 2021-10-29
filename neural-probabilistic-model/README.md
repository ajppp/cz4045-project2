# What I did (Jeth)
1. Download the wikitext-2 dataset

```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip
```

2. Get the starter code for the word language model

```bash
git clone git@github.com:pytorch/examples.git
mv ./examples/word_language_model ./
rm -rf examples
```

# Notes (Jeth)

## Dataset

WikiText-2 language modeling dataset.
100 million tokens extracted from the set of verified good and featured articles on Wiki. 
It has a large vocabulary and tetains the original case, punctuation and numbers
It was released by Salesforce in 2016 (really smth Joty would do... use his own company and colleague's work :P)
obtained from https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
Need to cite: https://arxiv.org/abs/1609.07843

## How the starter code works

There are different files so let's take a look from `main.py`

## Loading of data
This loading of data is done using `data.Corpus(path/to/data/)` (line 68 of main.py)
since we see that `Corpus` is a class defined in `data.py`, let us shift our attention to that file now.

## `data.py`

As can be seen in the file, when we initialise a `Corpus`,
we will be initialising a word dictionary (`Dictionary` class).
We will also be getting all the files of type train, test, valid which is present in the path. 
For each of the files, they will be tokenized by word (really 2016 approach instead of using subwords or bpe or spm)
after tokenization, they will be added to the word dictionary and given an index for each unique word.

The `Dictionary` class itself has its own `add_word` function which will ensure
that when adding the word, the word is not already present in the dictionary.
Thus, we know that there will only be one index per word.

After we have made the dictionary, we will then tokenize the files again,
this time, our aim is to convert all of the words into the vocab index as is present in the dictionary.
This gives us the index representation of the file which can be fed in into the model that we have.

## Next Steps

Since the dataset is large, we have to make it into batches so that it can fit
into the GPU. This is done conveniently for us using the `batchify` function in line 82-90 of `main.py`

Now we come to the actual modeling part. The authors already implemented a Transformer (Vaswani et al, 2017) 
as well as an RNN Model (Hochreiter and Schmidhuber (is it 1997 or 1996... i can't remember off of the top of my head))

## `model.py`

We look at how the RNN model is defined:
- `ntoken` refers to the number of tokens in the dataset i.e. size of the `Dictionary` that we made during preprocessing
- `ninp` refers to the size of each embedding vector i.e. the size of the embeddings that we generate from this input layer
- `nhid` refers to the number of neurons in the hidden layers (either tanh or relu)

Ignore Transformer since we are not looking at that (unfortunately)
Ignore PositionalEncoding since that is only used in Transformers to mitigate the fact that
the use of multi-headed self attention means that the Transformer is position invariant.
Hence, positional encoding is used so that the model have some sense of where the token is with 
respect to the other tokens. They used the vanilla encoding function of sine and cosine of diff frequencies.

# Implementation

We look at [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) by Bengio et al., 2003
A brief explanation of the model is given below (please actually read the paper... it's not really difficult to follow. thanks):

1. $(n - 1)$ word indexes are converted to word embeddings using the dictionary that we have. This is done on the embedding matrix $\mathbf{C}$. This gives us the distributed representation of the word in the real vector space, $V \in R^{ninp}$ (We have a parameter to control here `ninp`)
2. The input word features from the matrix $C$,  $(C(W_{t - n + 1}), \ldots, C(W_{t + 1}))$ are concatenated to give the word features layer activation vector, we call this vector $x$S
3. We then see that $X$ is passed to a hidden layer with tanh activation. (We have a parameter to control here!! size of this tanh layer)
4. the output of this layer would then be $tanh(d + Hx)$, $H$ is a $h \times (n - 1)m$ matrix where $h$ is the number of hidden units and $m$ the number of features associated with each word, $n - 1$, which i realise we have been using :P, is how many words to look at before the current token
. This output is then passed to another hidden layer with no bias vector i.e. simple matrix multiplication.
5. $x$ is also optionally passed to a separate hidden layer. This forms a skip connection and thus we get $Wx + b$ (if there are no connections, $W$ is just the 0 matrix)
6. The outputs are then added together to give us the equation $y = b + Wx + U \tanh(d + Hx)$ as seen in the paper. 
7. This output is the unnormalised log probabilities for each of the output word $i$.

Just as a summary, we have the parameters $\mathbf{\theta} = (b, d, W, U, H, C)$
$W$ is the matrix to convert the word feature to output weights (`ntoken` $\times ((n - 1) \cdot$ `ninp`))
$U$ is the matrix which contains the weight for the FC layer from the tanh to the output; dimension: `ntoken` $\times num neurons in tanh layer
$H$ already explained above, please look at step 4 lol
$b$ - bias term for the word feature to output weight
$d$ - bias term for the FC layer (tanh to output)
We also have optionally two different connections from the vector to the output.

At this point, i will create the model in FNNModel.py
