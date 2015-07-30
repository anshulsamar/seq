#seq
Sequence to Sequence RNN LSTM

Model from 'Sequence to Sequence Learning with Neural Networks' (Sutskever, et al. 2014). Note that the last encoder symbol is fed into the first decoder timestep as input. This model should be fairly adaptable to various sequence to sequence needs and encoder and decoder modules are built seperately with independent criterions. All data preprocessing is abstracted away - seq.lua assumes that 'data = dataLoader.load(opts)' loads necessary lookup tables, indexes, vocab sizes, sequence length information, and file names for pre-shuffled encoder and decoder data (which are read line by line).

Built ontop of github.com/wojzaremba/lstm

Some implementation notes:
- <EOS> and unknown words initialized to unit normal vectors
