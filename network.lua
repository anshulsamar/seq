require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'paths'
require 'math'
require 'EncCriterion'
require 'DecCriterion'
require 'utils/base'
require 'data'

-- wojrazemba lstm code https://github.com/wojzaremba/lstm/blob/master/main.lua

local function lstm(x, prev_c, prev_h, in_size, rnn_size)
   -- Calculate four gates together (rows of x are individual examples)
   local i2h = nn.Linear(in_size, 4*rnn_size)(x)
   local h2h = nn.Linear(rnn_size,4*rnn_size)(prev_h)
   local gates = nn.CAddTable()({i2h,h2h})

   -- Reshape to (batch_size, 4, rnn_size)
   -- Slice by gate, each row corresponds to example
   local reshaped_gates = nn.Reshape(4,rnn_size)(gates)
   local sliced_gates = nn.SplitTable(2)(reshaped_gates)

   -- LSTM memory cell
   local in_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
   local in_transform = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
   local forget_gate = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
   local out_gate = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

   local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
   local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

   return next_c, next_h
end

local function create_network(criterion,lookup,lookup_size,vocab_size,
                             in_size, rnn_size)
   local x = nn.Identity()()
   local y = nn.Identity()()
   local prev_s = nn.Identity()()
   LookupTable = nn.LookupTable(lookup_size,in_size)
   g_replace_table({LookupTable:parameters()[1]},{lookup})
   local i = {[0] = LookupTable(x)}
   local next_s = {}
   -- route elements of table to independent nodes
   local split = {prev_s:split(2 * opts.layers)}
   for layer_idx = 1, opts.layers do
      local prev_c = split[2 * layer_idx - 1]
      local prev_h = split[2 * layer_idx]
      local next_c, next_h
      if layer_idx == 1 then
         next_c, next_h = lstm(i[layer_idx-1], prev_c, prev_h, 
                                  in_size,rnn_size)
      else
         next_c, next_h = lstm(i[layer_idx-1], prev_c, prev_h, 
                                  rnn_size,rnn_size)
      end
      table.insert(next_s, next_c)
      table.insert(next_s, next_h)
      i[layer_idx] = next_h
   end
   local h2y = nn.Linear(rnn_size, vocab_size)
   local pred = nn.LogSoftMax()(h2y(i[opts.layers]))
   local err = criterion()({pred, y})
   local module = nn.gModule({x, y, prev_s},{err, nn.Identity()(next_s)})
   module:getParameters():uniform(-opts.weight_init, opts.weight_init)
   return g_transfer_data(module)

end

function setup_encoder()
   local net = create_network(EncCriterion,enc_data.lookup,
                            enc_data.lookup_size,enc_data.vocab_size,
                            opts.enc_in_size, opts.enc_rnn_size)
   encoder.x, encoder.dx = net:getParameters()
   encoder.net = g_cloneManyTimes(net, enc_data.len_max)
   encoder.s = {}
   encoder.ds = {}
   -- start from 0 as encoder.s[1] needs prev state too
   for j = 0, enc_data.len_max do
      encoder.s[j] = {}
      for d = 1, 2 * opts.layers do
         local outputStates = torch.zeros(opts.batch_size,opts.enc_rnn_size)
         encoder.s[j][d] = g_transfer_data(outputStates)
      end
   end
   for d = 1, 2 * opts.layers do
      local deltas = torch.zeros(opts.batch_size,opts.enc_rnn_size)
      encoder.ds[d] = g_transfer_data(deltas)
   end
   encoder.err = g_transfer_data(torch.zeros(enc_data.len_max))

   for d = 1, 2 * opts.layers do
      local out = torch.zeros(opts.batch_size,opts.enc_rnn_size)
      encoder.out[d] = g_transfer_data(out)
   end
   encoder.max_grad_norm = opts.max_grad_norm_enc
end

function setup_decoder()
   local net = create_network(DecCriterion,dec_data.lookup,
                            dec_data.lookup_size,dec_data.vocab_size,
                            opts.dec_in_size, opts.dec_rnn_size)
   decoder.x, decoder.dx = net:getParameters()
   decoder.net = g_cloneManyTimes(net, dec_data.len_max)
   decoder.s = {}
   decoder.ds = {}
   for j = 0, dec_data.len_max do
      decoder.s[j] = {}
      for d = 1, 2 * opts.layers do
         local outputStates = torch.zeros(opts.batch_size,opts.dec_rnn_size)
         decoder.s[j][d] = g_transfer_data(outputStates)
      end
   end
   for d = 1, 2 * opts.layers do
      local deltas = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      decoder.ds[d] = g_transfer_data(deltas)
   end
   decoder.err = g_transfer_data(torch.zeros(dec_data.len_max))
   decoder.max_grad_norm = opts.max_grad_norm_dec
end

function setup_mlp()
   for d = 1, 2 * opts.layers do
      local h1 = nn.Identity()()
      local h2 = nn.Tanh()(nn.Linear(opts.enc_rnn_size, opts.dec_rnn_size)(h1))
      local net = nn.gModule({h1}, {h2})
      net:getParameters():uniform(-opts.weight_init, opts.weight_init)
      mlp.mu.net[d] = g_transfer_data(net)
   end

   for d = 1, 2 * opts.layers do
      local s = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      mlp.mu.s[d] = g_transfer_data(s)
      local ds = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      mlp.mu.ds[d] = g_transfer_data(ds)
   end

   for d = 1, 2 * opts.layers do
      local h1 = nn.Identity()()
      local h2 = nn.Tanh()(nn.Linear(opts.enc_rnn_size, opts.dec_rnn_size)(h1))
      local net = nn.gModule({h1}, {h2})
      net:getParameters():uniform(-opts.weight_init, opts.weight_init)
      mlp.lsigs.net[d] = g_transfer_data(net)
   end

   for d = 1, 2 * opts.layers do
      local s = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      mlp.lsigs.s[d] = g_transfer_data(s)
      local ds = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      mlp.lsigs.ds[d] = g_transfer_data(ds)
   end

   mlp.max_grad_norm = opts.max_grad_norm_mlp
end
