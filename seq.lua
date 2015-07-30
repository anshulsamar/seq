-- Copyright (c) 2015 Anshul Samar
-- Copyright (c) 2014, Facebook, Inc. All rights reserved.
-- Licensed under the Apache License, Version 2.0
-- See original code: github.com/wojzaremba/lstm

require 'io'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'EncoderCriterion'
require 'DecoderCriterion'
require 'base'
require 'dataLoader'

local function transfer_data(x)
   return x:cuda()
end

local encoder, decoder
local model = {}
local params = {encoderx ={}, encoderdx = {}, decoderx = {}, decoderdx = {}}
local data = {}

local function lstm(x, prev_c, prev_h)
   -- Calculate four gates together (rows of x are individual examples)
   local i2h = nn.Linear(opts.in_size, 4*opts.rnn_size)(x)
   local h2h = nn.Linear(opts.rnn_size,4*opts.rnn_size)(prev_h)
   local gates = nn.CAddTable()({i2h,h2h})

   -- Reshape to (batch_size, 4, opts.rnn_size)
   -- Slice by gate, each row corresponds to example
   local reshaped_gates = nn.Reshape(4,opts.rnn_size)(gates)
   local sliced_gates = nn.SplitTable(2)(reshaped_gates)

   -- LSTM memory cell
   local in_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
   local in_trainsform = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
   local forget_gate = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
   local out_gate = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

   local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}),
                                  nn.CMulTable()({in_gate, in_transform})})
   local next_h = nn.CMulTable(){out_gate,nn.Tanh()(next_c)}
   return next_c, next_h
end

local function create_network(criterion,lookup,vocab_size)
   local x = nn.Identity()()
   local y = nn.Identity()()
   local prev_s = nn.Identity()()
   LookupTable = nn.LookupTable(vocab_size,opts.in_size)
   LookupTable:parameters()[1] = lookup
   local i = {[0] = LookupTable(x)}
   local next_s = {}
   -- route elements of table to independent nodes
   local split = {prev_s:split(2 * opts.layers)}
   for layer_idx = 1, opts.layers do
      local prev_c = split[2 * layer_idx - 1]
      local prev_h = split[2 * layer_idx]
      local next_c, next_h = lstm(x, prev_c, prev_h)
      table.insert(next_s, next_c)
      table.insert(next_s, next_h)
      i[layer_idx] = next_h
   end
   local h2y = nn.Linear(opts.rnn_size, vocab_size)
   local pred = nn.LogSoftMax()(h2y)
   local err = criterion()({pred, y})
   local module = nn.gModule({x, y, prev_s},{err, nn.Identity()(next_s)})
   module:getParameters():uniform(-opts.weight_init, opts.weight_init)
   return transfer_data(module)
end

local function setupEncoder()
   encoder = create_network(EncoderCriterion,data.enc_lookup,data.enc_vocab_size)
   params.encoderx, params.encoderdx = encoder:getParameters()
   model.encoder = g_cloneManyTimes(encoder, data.enc_len_max)
   model.enc_s = {}
   model.enc_ds = {}
   for j = 0, data.enc_len_max do
      model.enc_s[j] = {}
      for d = 1, 2 * opts.layers do
         local outputStates = torch.zeros(opts.batch_size,opts.rnn_size)
         model.enc_s[j][d] = transfer_data(outputStates)
      end
   end
   for d = 1, 2 * opts.layers do
      local deltas = torch.zeros(opts.batch_size,opts.rnn_size)
      model.enc_ds[d] = transfer_data(deltas)
   end
   model.dec_norm_dw = 0
   model.dec_err = transfer_data(torch.zeros(data.dec_len_max))
end

local function setupDecoder()
   decoder = create_network(DecoderCriterion,data.dec_lookup,data.dec_vocab_size)
   params.decoderx, params.decoderdx = decoder:getParameters()
   model.decoder = g_cloneManyTimes(decoder, data.dec_len_max)
   model.dec_s = {}
   model.dec_ds = {}
   for j = 0, data.dec_len_max do
      model.dec_s[j] = {}
      for d = 1, 2 * opts.layers do
         local outputStates = torch.zeros(opts.batch_size,opts.rnn_size)
         model.dec_s[j][d] = transfer_data(outputStates)
      end
   end
   for d = 1, 2 * opts.layers do
      local deltas = torch.zeros(opts.batch_size,opts.rnn_size)
      model.dec_ds[d] = transfer_data(deltas)
   end
   model.dec_norm_dw = 0
   model.dec_err = transfer_data(torch.zeros(data.dec_len_max))
end

local function reset_s()
   for j = 0, data.enc_len_max do
      for d = 1, 2 * opts.layers do
         model.enc_s[j][d]:zero()
      end
   end

   for j = 0, data.dec_len_max do
      for d = 1, 2 * opts.layers do
         model.dec_s[j][d]:zero()
      end
   end
end

local function reset_ds()
   for d = 1, 2 * opts.layers do
      model.enc_ds[d]:zero()
      model.dec_ds[d]:zero()
   end
end

local function fp(enc_x, enc_y, dec_x, dec_y)
   reset_s()
   local ret
   for i = 1, batch.enc_len_seq do
      local s = model.enc_s[i - 1]
      ret = model.encoder[i]:forward({enc_x[i], enc_y[i], s})
      model.enc_error[i] = ret[1]
      model.enc_s[i] = ret[2]
   end
   model.dec_s[0] = model.enc_s[batch.enc_len_seq]
   for i = 1, batch.dec_len_seq do
      local s = model.dec_s[i - 1]
      ret = model.decoder[i]:forward({dec_x[i], dec_y[i], s})
      model.dec_error[i] = ret[1]
      model.dec_s[i] = ret[2]
   end
end

local function bp(enc_x,enc_y,dec_x,dec_y)
   params.encoderdx:zero()
   params.decoderdx:zero()
   reset_ds()

   for i = batch.dec_len_seq, 1, -1 do
      local s = model.dec_s[i-1]
      local derr = transfer_data(torch.ones(1))
      local input = {dec_x[i], dec_y[i], s}
      local output = {derr, model.dec_ds}
      local tmp = model.decoder[i]:backward(input, output)[3]
      g_replace_table(model.dec_ds, tmp)
      cutorch.synchronize()
   end

   g_replace_table(model.enc_ds,model.dec_ds)

   for i = batch.enc_seq_len, 1, -1 do
      local s = model.enc_s[i-1]
      local derr = transfer_data(torch.ones(1))
      local input = {enc_x[i], enc_y[i], s}
      local output = {derr, model.enc_ds}
      local tmp = model.encoder[i]:backward(input, output)[3]
      g_replace_table(model.enc_ds, tmp)
      cutorch.synchronize()
   end

   model.enc_norm_dw = params.encoderdx:norm()
   if model.enc_norm_dw > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/model.enc_norm_dw
      params.encoderdx:mul(shrink_factor)
   end
   params.encoderx:add(params.encoderdx:mul(-opts.lr))

   model.dec_norm_dw = params.decoderdx:norm()
   if model.dec_norm_dw > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/model.dec_norm_dw
      params.decoderdx:mul(shrink_factor)
   end
   params.decoderx:add(params.decoderdx:mul(-opts.lr))
end

local function run()
   g_init_gpu(1)
   local cmd = torch.CmdLine()
   cmd:option('-layers',2)
   cmd:option('-in_size',300)
   cmd:option('-rnn_size',300)
   cmd:option('-batch_size',128)
   cmd:option('-max_grad_norm',5)
   cmd:option('-epochs',7)
   cmd:option('-anneal_after',5)
   cmd:option('-anneal_every',1)
   cmd:option('-decay',2)
   cmd:option('-weight_init',.08)
   cmd:option('-lr',0.7)
   cmd:option('-run_dir','./exp/')
   opts = cmd:parse(arg)

   if paths.dir(opts.run_dir) == nil then
      paths.mkdir(opts.run_dir)
   end

   print("Saving Options")
   torch.save(paths.concat(opts.run_dir,'opts.th7'),opts)

   print("Loading Data")
   data = dataLoader.load()

   print("Creating network")
   setupEncoder()
   setupDecoder()

   local beginning_time = torch.tic()
   local start_time = torch.tic()
   
   print("Starting training")
   
   for epoch=1,opts.epochs do
      print('Epoch ' .. epoch)
      if epoch > opts.anneal_epoch then
         opts.lr = opts.lr / opts.decay
      end
      
      local iteration = 0
      local enc_f = io.open(data.encoder_data_file,'r')
      local dec_f = io.open(data.decoder_data_file,'r')
      while true do
         iteration += 1
         collectgarbage()
         local num_lines = 0
         local enc_line = {}
         local dec_line = {}
         while true do
            table.insert(enc_line,io:read("*l"))
            table.insert(dec_line = io:read("*l"))
            if enc_line == nil or dec_line == nil then
               break
            end
            num_lines += 1
            if num_lines == opts.batch_size then
               break
            end
         end
         local curr_batch_size = num_lines
         local enc_x = {}
         local enc_y = {}
         for i=1,opts.enc_len_max do
            table.insert(enc_x,torch.zeros(curr_batch_size))
            table.insert(enc_y,torch.zeros(curr_batch_size))
         end
         local dec_x = {}
         local dec_y = {}
         for i=1,opts.dec_len_max do
            table.insert(dec_x,torch.zeros(curr_batch_size))
            table.insert(dec_y,torch.zeros(curr_batch_size))
         end
         local enc_len_seq = 0
         local dec_len_seq = 0

         for i=1,num_lines do
            local num_word = 1
            for enc_word in stringx.split(enc_line[i],' ') do
               if enc_word ~= "" then
                  enc_x[num_line][num_word] = data.enc_index[enc_word]
                  num_word += 1
               end
            end

            if num_word > enc_len_seq then
               enc_len_seq = num_word
            end
            
            dec_x[num_line][1] = data.enc_index[enc_word]

            local num_word = 1
            for dec_word in stringx.split(dec_line[i],' ') do
               dec_y[num_line][num_word] = data.dec_index[enc_word]
               if num_word < opts.dec_len_max then
                  dec_x[num_line][num_word+1] = data.dec_index[enc_word]
               end
               num_word += 1
            end

            table.remove(dec_x)

            if num_word > dec_len_seq then
               dec_len_seq = num_word
            end
         end

         opts.enc_len_seq = enc_len_seq
         opts.dec_len_seq = dec_len_seq

         fp(enc_x,enc_y,dec_x,dec_y)
         bp(enc_x,enc_y,dec_x,dec_y)

         if curr_batch_size ~= opts.batch_size then
            break
         end

         if iteration % 33 == 0 then
            cutorch.synchronize()
         end

         print('epoch = ' .. g_f3(epoch) ..
               'iteration = ' .. g_f3(iteration) ..
               'meanError = ' .. err ..
               'encNorm = ' .. g_f3(model.enc_norm_dw) ..
               'decNorm = ' .. g_f3(model.dec_norm_dw) ..
               'lr = ' ..  g_f3(params.lr) ..
               'time = ' .. since_beginning .. ' mins.')

      end
      enc_f:close()
      dec_f:close()
   end         
end


   



