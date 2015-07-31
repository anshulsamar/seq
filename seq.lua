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
require 'EncCriterion'
require 'DecCriterion'
require 'base'
require 'dataLoader'
require 'reload'
require 'paths'

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

local function create_network(criterion,lookup,lookup_size,vocab_size)
   local x = nn.Identity()()
   local y = nn.Identity()()
   local prev_s = nn.Identity()()
   LookupTable = nn.LookupTable(lookup_size,opts.in_size)
   g_replace_table({LookupTable:parameters()[1]},{lookup})
   local i = {[0] = LookupTable(x)}
   local next_s = {}
   -- route elements of table to independent nodes
   local split = {prev_s:split(2 * opts.layers)}
   for layer_idx = 1, opts.layers do
      local prev_c = split[2 * layer_idx - 1]
      local prev_h = split[2 * layer_idx]
      local next_c, next_h = lstm(i[layer_idx-1], prev_c, prev_h)
      table.insert(next_s, next_c)
      table.insert(next_s, next_h)
      i[layer_idx] = next_h
   end
   local h2y = nn.Linear(opts.rnn_size, vocab_size)
   local pred = nn.LogSoftMax()(h2y(i[opts.layers]))
   local err = criterion()({pred, y})
   local module = nn.gModule({x, y, prev_s},{err, nn.Identity()(next_s)})
   module:getParameters():uniform(-opts.weight_init, opts.weight_init)
   return transfer_data(module)
end

local function setupEncoder()
   encoder = create_network(EncCriterion,enc_data.lookup,enc_data.lookup_size,enc_data.vocab_size)
   params.encoderx, params.encoderdx = encoder:getParameters()
   model.encoder = g_cloneManyTimes(encoder, enc_data.len_max)
   model.enc_s = {}
   model.enc_ds = {}
   for j = 0, enc_data.len_max do
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
   model.enc_norm_dw = 0
   model.enc_err = transfer_data(torch.zeros(enc_data.len_max))
end

local function setupDecoder()
   decoder = create_network(DecCriterion,dec_data.lookup,dec_data.lookup_size,dec_data.vocab_size)
   params.decoderx, params.decoderdx = decoder:getParameters()
   model.decoder = g_cloneManyTimes(decoder, dec_data.len_max)
   model.dec_s = {}
   model.dec_ds = {}
   for j = 0, dec_data.len_max do
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
   model.dec_err = transfer_data(torch.zeros(dec_data.len_max))
end

local function reset_s()
   for j = 0, enc_data.len_max do
      for d = 1, 2 * opts.layers do
         model.enc_s[j][d]:zero()
      end
   end

   for j = 0, dec_data.len_max do
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

local function fp(enc_x, enc_y, dec_x, dec_y, batch)
   reset_s()
   local ret
   for i = 1, batch.enc_len_seq do
      local s = model.enc_s[i - 1]
      ret = model.encoder[i]:forward({enc_x[i], enc_y[i], s})
      model.enc_err[i] = ret[1]
      model.enc_s[i] = ret[2]
   end

   --for i = 1, batch.curr_batch_size do
   --model.dec_s[0][i] = model.enc_s[batch.enc_lengths[i]][i]
   --end

   model.dec_s[0] = model.enc_s[batch.enc_len_seq]
   for i = 1, batch.dec_len_seq do
      local s = model.dec_s[i - 1]
      ret = model.decoder[i]:forward({dec_x[i], dec_y[i], s})
      model.dec_err[i] = ret[1]
      model.dec_s[i] = ret[2]
   end
end

local function bp(enc_x,enc_y,dec_x,dec_y, batch)
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

   for i = 0, batch.enc_len_seq-1 do
      local new_i = batch.enc_len_seq - i
      local s = model.enc_s[new_i-1]
      local derr = transfer_data(torch.ones(1))
      local input = {enc_x[new_i], enc_y[new_i], s}
      local output = {derr, model.enc_ds}
      local b = model.encoder[new_i]:backward(input, output)
      local tmp = b[3]
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


-- group things together that are same size
-- CHANGE BATCH SIZE? BUG?
-- CHANGE CRITERION BACK TO ENC/DEC
-- ENC CRITERION RETURNS 0 instead of vector - bug?
-- check what happens if batch size is less or more or less examples than batch_size
-- currently default for decy ency is 1
-- currently made default for 0 vector input
-- how to do mini batch with sequence (Because even the 0 input vector gets propogated due to hidden state) 
-- make mini batches similar size
-- print out model.enc_norm stuff using gf3 and since begginning
-- print out errors too

function run()
   print("\27[31mStaring Experiment\n---------------")
   g_init_gpu({1})
   local cmd = torch.CmdLine()
   cmd:option('-layers',2)
   cmd:option('-in_size',300)
   cmd:option('-rnn_size',300)
   cmd:option('-batch_size',2)
   cmd:option('-max_grad_norm',5)
   cmd:option('-epochs',7)
   cmd:option('-anneal_after',5)
   cmd:option('-decay',2)
   cmd:option('-weight_init',.08)
   cmd:option('-lr',0.7)
   cmd:option('-run_dir','./exp/')
   cmd:option('-save_dir','./exp/model/')
   cmd:option('-load_model',false)
   opts = cmd:parse(arg)

   if paths.dir(opts.run_dir) == nil then
      paths.mkdir(opts.run_dir)
      paths.mkdir(opts.run_dir .. '/data/')
      paths.mkdir(opts.run_dir .. '/model/')
      paths.mkdir(opts.run_dir .. '/log/')
   end
   print(opts)
   print("Saving Options")
   torch.save(paths.concat(opts.run_dir,'opts.th7'),opts)

   print("Loading Data")
   enc_data, dec_data = dataLoader.get()

   print("\27[31mCreating Network\n----------------")
   print("Setting up Encoder")
   setupEncoder()
   print("Setting up Decoder")
   setupDecoder()
   local start = 1

   if opts.load_model then
      print("Loading previous parameters")
      oldModel = torch.load(opts.save_dir .. '/model.th7')
      g_replace_table(params.encoderx,oldModel[1].encoderx)
      g_replace_table(params.encoderdx,oldModel[1].encoderdx)
      g_replace_table(params.decoderx,oldModel[1].decoderx)
      g_replace_table(params.decoderdx,oldModel[1].decoderdx)
      start = oldModel[2]
      opts.lr = oldModel[3]
   end

   local beginning_time = torch.tic()
   local start_time = torch.tic()
   
   print("\27[31mTraining\n----------")
   
   for epoch=start,opts.epochs do
      print('Epoch ' .. epoch)
      if epoch > opts.anneal_after then
         opts.lr = opts.lr / opts.decay
      end
      
      local iteration = 0
      local enc_f = io.open(enc_data.file_path,'r')
      local dec_f = io.open(dec_data.file_path,'r')
      while true do
         iteration = iteration + 1
         local num_lines = 0
         local enc_line = {}
         local dec_line = {}
         while true do
            table.insert(enc_line,enc_f:read("*l"))
            table.insert(dec_line,dec_f:read("*l"))
            if enc_line[#enc_line] == nil or dec_line[#dec_line] == nil then
               break
            end
            num_lines = num_lines + 1
            if num_lines == opts.batch_size then
               break
            end
         end
         local curr_batch_size = num_lines

         if curr_batch_size == 0 then break end

         local enc_x = {}
         local enc_y = {}
         local batch = {}
         batch.curr_batch_size = curr_batch_size
         batch.enc_lengths = torch.zeros(curr_batch_size)
         batch.dec_lengths = torch.zeros(curr_batch_size)
         for i=1,enc_data.len_max do
            local x_init = torch.ones(curr_batch_size) * enc_data.default_index
            table.insert(enc_x,transfer_data(x_init))
            table.insert(enc_y,transfer_data(torch.zeros(curr_batch_size)))
         end
         local dec_x = {}
         local dec_y = {}
         for i=1,dec_data.len_max do
            local x_init = torch.ones(curr_batch_size) * dec_data.default_index
            table.insert(dec_x,transfer_data(x_init))
            table.insert(dec_y,transfer_data(torch.zeros(curr_batch_size)))
         end
         local enc_len_seq = 0
         local dec_len_seq = 0

         for i=1,num_lines do
            if enc_line[i] ~= nil then
               local num_word = 0
               local last_word = ""
               local len = 0
               for _,enc_word in ipairs(stringx.split(enc_line[i],' ')) do
                  if enc_word ~= "" and len < enc_data.len_max  then
                     len = len + 1
                  end
               end
               local offset = enc_data.len_max - len
               for _,enc_word in ipairs(stringx.split(enc_line[i],' ')) do
                  if enc_word ~= "" and num_word < enc_data.len_max  then
                     num_word = num_word + 1
                     enc_x[num_word + offset][i] = enc_data.index[enc_word]
                     last_word = enc_word
                  end
               end

               if num_word > enc_len_seq then
                  enc_len_seq = num_word
               end

               batch.enc_lengths[i] = num_word
               
               dec_x[1][i] = enc_data.index[last_word]

               local num_word = 0
               local indexes = {}
               for _,dec_word in ipairs(stringx.split(dec_line[i],' ')) do
                  if dec_word ~= "" and num_word < dec_data.len_max then
                     num_word = num_word + 1
                     dec_y[num_word][i] = dec_data.index[dec_word]
                     indexes[num_word] = {dec_data.index[dec_word],dec_word}
                  end
               end

               for j=1,#indexes - 1 do
                  dec_x[j+1][i] = indexes[j][1]
               end

               if num_word > dec_len_seq then
                  dec_len_seq = num_word
               end

               batch.dec_lengths[i] = num_word

            end
         end

         batch.enc_len_seq = enc_len_seq
         batch.dec_len_seq = dec_len_seq

         print("Forward")
         fp(enc_x,enc_y,dec_x,dec_y,batch)
         print("Backward")
         bp(enc_x,enc_y,dec_x,dec_y,batch)

         if curr_batch_size ~= opts.batch_size then
            break
         end

         if iteration % 33 == 0 then
            cutorch.synchronize()
         end

         print('epoch = ' .. epoch .. 'iteration = ' .. iteration .. 'lr = ' ..  opts.lr)

      end
      print('Finished epoch')
      enc_f:close()
      dec_f:close()
      torch.save(opts.save_dir .. '/model.th7', {params, epochs, opts.lr})
   end         
end


   



