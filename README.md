#seq
Sequence to Sequence RNN LSTM

Sequence to Sequence RNN LSTM model built upon Wojzaremba's LSTM and LM code (github.com/wojzaremba/lstm). The model is based on the 2014 NIPS paper 'Sequence to Sequence Learning with Neural Networks' (Sutskever, et al) and can be used for NLP tasks including machine translation and sentence embeddings.

Note that during training, the correct target sequence is fed into the decoder. For example, for the sentence 'cold weather', 'cold' is fed as input during the timestep the RNN attempts to predict 'weather.' 

Note that the last encoder symbol is fed into the first decoder timestep as input. This model should be fairly adaptable to various sequence to sequence needs as encoder and decoder modules are built seperately with independent criterions. All data preprocessing is abstracted away - seq.lua assumes that 'data = dataLoader.load(opts)' loads necessary lookup tables, indexes, vocab sizes, sequence length information, and file names for pre-shuffled encoder and decoder data (which are read line by line). For an example dataLoader, see dataLoader.lua (meant for word embeddings where encoder and decoder files are the same).

Built ontop of github.com/wojzaremba/lstm

Some implementation notes:
- EOS and unknown words initialized to unit normal vectors
- Currently data is shuffled but not sorted by length


Note, currently no dropout support (although it should be fairly easy to re-add Wojzaremba's dropout code. I removed it for now, because I hadn't integrated it). I removed or altered other functions I was not using as well.
