# End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial

This is a PyTorch tutorial for the ACL'16 paper 
[**End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF**](http://www.aclweb.org/anthology/P16-1101)

This repository includes

* [**IPython Notebook of the tutorial**](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb)
* Data folder
* Setup Instructions file
* Pretrained models directory (The notebook will automatically download pre-trained models into this directory, as required)

### Authors

[**Anirudh Ganesh**](https://github.com/TheAnig)

[**Peddamail Jayavardhan Reddy**](https://github.com/jayavardhanr)


### Installation
The best way to install pytorch is via the [**pytorch webpage**](http://pytorch.org/)

### Setup

#### Creating new Conda environment
`conda create -n pytorch python=3.5`

#### Activate the condo environment
`source activate pytorch`

#### Setting up notebooks with specific python version (python 3.5)
```
conda install notebook ipykernel
ipython kernel install --user
```

#### PyTorch Installation command:
`conda install pytorch torchvision -c pytorch`

#### NumPy installation
`conda install -c anaconda numpy`

#### Download GloVe vectors and extract glove.6B.100d.txt into "./data/" folder

`wget http://nlp.stanford.edu/data/glove.6B.zip`

#### Data Files

You can download the data files from within this repo [**over here**](https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data)

## Jeth

Okay, the model consists of two parts:
- character level encoder implemented using either a CNN or an LSTM
- word level encoder for which we will build a CNN using max pooling


In the model definition, they gave the code for the character level encoder (no need to change)
    ```python
    self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
    init_embedding(self.char_embeds.weight)
    ```

The layer is implemented in the function `get_lstm_features`

```python
chars_embeds = chars_embeds = self.char_embeds(chars2).unsqueeze(1)
```


Then a Conv layer is added to generate the spatial coherence across the characters (back to model definition)
```python
self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))
```


The character embedding from the previous layer is then passed here in `get_lstm_features`:
```python
chars_cnn_out3 = self.char_cnn3(chars_embeds)
```

We then pass it through a max pool layer
```python
chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                             kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)
```

What we have now is a vector representation of the word from the character representation. We then concatenate the representation with the GloVe embeddings:

```python
embeds = self.word_embeds(sentence)
embeds = torch.cat((embeds, chars_embeds), 1)
```
