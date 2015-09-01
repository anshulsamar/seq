#seq
Sequence to Sequence RNN LSTM

Model from 'Sequence to Sequence Learning with Neural Networks' (Sutskever, et al. 2014). Note that the last encoder symbol is fed into the first decoder timestep as input. This model should be fairly adaptable to various sequence to sequence needs as encoder and decoder modules are built seperately with independent criterions. All data preprocessing is abstracted away - seq.lua assumes that 'data = dataLoader.load(opts)' loads necessary lookup tables, indexes, vocab sizes, sequence length information, and file names for pre-shuffled encoder and decoder data (which are read line by line). For an example dataLoader, see dataLoader.lua (meant for word embeddings where encoder and decoder files are the same).

Built ontop of github.com/wojzaremba/lstm

Some implementation notes:
- EOS and unknown words initialized to unit normal vectors
- Currently data is shuffled but not sorted by length


Note, currently no dropout support (although it should be fairly easy to re-add Wojzaremba's dropout code. I removed it for now, because I hadn't integrated it). I removed other functions I was not using as well.
