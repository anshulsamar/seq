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
require 'gnuplot'
require 'math'

local function transfer_data(x)
   return x:cuda()
end

model = {}
local encoder, decoder
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
   encoder = create_network(EncCriterion,enc_data.lookup,
                            enc_data.lookup_size,enc_data.vocab_size)
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
   decoder = create_network(DecCriterion,dec_data.lookup,
                            dec_data.lookup_size,dec_data.vocab_size)
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
   for i = 1, batch.enc_len_max do
      local s = model.enc_s[i - 1]
      ret = model.encoder[i]:forward({enc_x[i], enc_y[i], s})
      model.enc_err[i] = ret[1]
      model.enc_s[i] = ret[2]
   end

   --for i = 1, batch.size do
   --model.dec_s[0][i] = model.enc_s[batch.enc_lengths[i]][i]
   --end

   model.dec_s[0] = model.enc_s[batch.enc_len_max]
   for i = 1, batch.dec_len_max do
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

   for i = batch.dec_len_max, 1, -1 do
      local s = model.dec_s[i-1]
      local derr = transfer_data(torch.ones(1))
      local input = {dec_x[i], dec_y[i], s}
      local output = {derr, model.dec_ds}
      local tmp = model.decoder[i]:backward(input, output)[3]
      g_replace_table(model.dec_ds, tmp)
      cutorch.synchronize()
   end

   g_replace_table(model.enc_ds,model.dec_ds)

   for i = 0, batch.enc_len_max-1 do
      local new_i = batch.enc_len_max - i
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


local function getError()
   local tot_enc_err = 0
   for i = 1, batch.enc_len_max do
      tot_enc_err = tot_enc_err + model.enc_err[i]
   end
   tot_enc_err = tot_enc_err * opts.batch_size / batch.size

   local tot_dec_err = 0
   for i = 1, batch.dec_len_max do
      tot_dec_err = tot_dec_err + model.dec_err[i]
   end
   
   tot_dec_err = tot_dec_err * opts.batch_size / batch.size
   return tot_enc_err, tot_dec_err
end


local function log(epoch, iter, max_iter)
   stats.enc_err, stats.dec_err = getError()
   stats.avg_enc_err = ((stats.avg_enc_err * (iter-1)) + stats.enc_err)/iter
   stats.avg_dec_err = ((stats.avg_dec_err * (iter-1)) + stats.dec_err)/iter
   print('epoch=' .. string.format('%02d',epoch + opts.start) ..  
         ', iter=' .. string.format('%03d',iter) ..
         ', enc_err=' .. string.format('%.2f',stats.enc_err) ..
         ', avg_enc_err=' .. string.format('%.2f',stats.avg_enc_err) ..
         ', dec_err=' .. string.format('%.2f',stats.dec_err) .. 
         ', avg_dec_err=' .. string.format('%.2f',stats.avg_dec_err) ..
         ', encdxNorm=' .. string.format('%.4f',model.enc_norm_dw) ..
         ', decdxNorm=' .. string.format('%.4f',model.dec_norm_dw) .. 
         ', lr=' ..  string.format('%.3f',opts.lr))


   local logName = opts.log_dir .. 'log.txt'
   if iter == max_iter then
      table.insert(stats.avg_dec_err_epoch,stats.avg_dec_err)
   end
end

local function makeDirectories()
   if paths.dir(opts.run_dir) == nil then
      paths.mkdir(opts.run_dir)
      paths.mkdir(opts.data_dir_to)
      paths.mkdir(opts.decode_dir)
      paths.mkdir(opts.save_dir)
      paths.mkdir(opts.log_dir)
   end
end

local function loadModel()
   filen = opts.save_dir .. '/model.th7'
   if (paths.filep(filen)) then
      print("Loading previous parameters")
      local oldModel = torch.load(opts.save_dir .. '/model.th7')
      params.encoderx:copy(oldModel[1].encoderx)
      params.encoderdx:copy(oldModel[1].encoderdx)
      params.decoderx:copy(oldModel[1].decoderx)
      params.decoderdx:copy(oldModel[1].decoderdx)
      opts.start = oldModel[2]
      opts.lr = oldModel[3]
      stats = oldModel[4]
   else
      print('No model to load, training from scratch')
   end
end

function plotErr(modelFile)
   if stats == nil then
      local oldModel = torch.load(modelFile)
      stats = oldModel[4]
   end

   gnuplot.plot(torch.Tensor(stats.avg_dec_err_epoch))
   gnuplot.title('Average Decoder Error vs Epochs')
   gnuplot.xlabel('Epoch')
   gnuplot.ylabel('Negative Log Likelihood')
end


local function initializeBatch(size)
   batch.size = size
   batch.enc_lengths = torch.zeros(opts.batch_size)
   batch.dec_lengths = torch.zeros(opts.batch_size)
end

local function initializeEncMat()
   for i=1,enc_data.len_max do
      local x_init = torch.ones(opts.batch_size) * enc_data.default_index
      table.insert(enc_x,transfer_data(x_init))
      table.insert(enc_y,transfer_data(torch.zeros(opts.batch_size)))
   end
end

local function initializeDecMat()
   for i=1,dec_data.len_max do
      local x_init = torch.ones(opts.batch_size) * dec_data.default_index
      table.insert(dec_x,transfer_data(x_init))
      table.insert(dec_y,transfer_data(torch.zeros(opts.batch_size)))
   end
end

local function loadMat(encLine,decLine,i)

   local enc_num_word = 0
   local last_word = ""
   local len = 0
   for _,enc_word in ipairs(stringx.split(encLine,' ')) do
      if enc_word ~= "" and len < enc_data.len_max  then
         len = len + 1
      end
   end
   local offset = enc_data.len_max - len
   for _,enc_word in ipairs(stringx.split(encLine,' ')) do
      if enc_word ~= "" and enc_num_word < enc_data.len_max  then
         enc_num_word = enc_num_word + 1
         enc_x[enc_num_word + offset][i] = enc_data.index[enc_word]
         last_word = enc_word
      end
   end

   batch.enc_lengths[i] = enc_num_word
   
   dec_x[1][i] = enc_data.index[last_word]

   local dec_num_word = 0
   local indexes = {}
   for _,dec_word in ipairs(stringx.split(decLine,' ')) do
      if dec_word ~= "" and dec_num_word < dec_data.len_max then
         dec_num_word = dec_num_word + 1
         dec_y[dec_num_word][i] = dec_data.index[dec_word]
         indexes[dec_num_word] = {dec_data.index[dec_word],dec_word}
      end
   end

   for j=1,#indexes - 1 do
      dec_x[j+1][i] = indexes[j][1]
   end

   batch.dec_lengths[i] = dec_num_word

   return enc_num_word, dec_num_word
end

local function decode(epoch,iter,batch,dec_line)

   local indexes = {}
   for i=1,#model.output do
      local y, ind = torch.max(model.output[i],2)
      table.insert(indexes,ind)
   end

   local decodeName = 'decode_' .. epoch .. '_' .. iter .. '.txt'
   local f = io.open(opts.decode_dir .. decodeName,'a+')

   for i=1,#dec_line do
      local num_words = batch.dec_line_length[i]
      local sentence = ''
      for j=1,num_words do
         sentence = sentence .. dec_data.rev_index[indexes[j][i][1]] .. ' '
      end
      f:write(sentence,'\n')
      f:flush()
   end

   f:close()

end

local function getOpts()
   local cmd = torch.CmdLine()
   cmd:option('-layers',2)
   cmd:option('-in_size',300)
   cmd:option('-rnn_size',300)
   cmd:option('-batch_size',1)
   cmd:option('-max_grad_norm',5)
   cmd:option('-max_epoch',10)
   cmd:option('-start',0)
   cmd:option('-anneal',false)
   cmd:option('-anneal_after',4)
   cmd:option('-decay',2)
   cmd:option('-weight_init',.1)
   cmd:option('-lr',.7)
   cmd:option('-data_dir_from','/deep/group/speech/asamar/nlp/data/gutenberg/txt/')
   cmd:option('-base_path','/deep/group/speech/asamar/nlp/seq/')
   cmd:option('-glove_path','/deep/group/speech/asamar/nlp/glove/pretrained/glove.840B.300d.txt')
   cmd:option('-run_dir','/deep/group/speech/asamar/nlp/seq/exp/')
   cmd:option('-load_model',false)
   cmd:option('-parser','Gut')
   local opts = cmd:parse(arg)
   opts.decode_dir = opts.run_dir .. '/decode/'
   opts.data_dir_to = opts.run_dir .. '/data/'
   opts.save_dir = opts.run_dir .. '/model/'
   opts.log_dir = opts.run_dir .. '/log/'

   return opts
end

function run()
   -- Options
   print("\27[31mStarting Experiment\n---------------")
   g_init_gpu({1})
   opts = getOpts()
   makeDirectories()
   print(opts)
   print("Saving Options")
   torch.save(paths.concat(opts.run_dir,'opts.th7'),opts)

   -- Data
   print("Loading Data")
   enc_data, dec_data = dataLoader.get(opts)
   local max_iter = math.ceil(enc_data.total_lines/opts.batch_size)

   -- Network
   print("\27[31mCreating Network\n----------------")
   print("Setting up Encoder")
   setupEncoder()
   print("Setting up Decoder")
   setupDecoder()
   if opts.load_model then loadModel() end

   -- Training
   print("\27[31mTraining\n----------")
   stats = {}
   stats.avg_dec_err_epoch = {}

   for epoch=1,(opts.max_epoch - opts.start) do

      -- Setup
      local iter = 0

      stats.avg_enc_err = 0
      stats.avg_dec_err = 0
      stats.dec_err = 0
      stats.enc_err = 0

      -- Anneal
      if opts.anneal and (epoch + opts.start) > opts.anneal_after then
         opts.lr = opts.lr / opts.decay
      end

      -- Open Data
      local enc_f = io.open(enc_data.file_path,'r')
      local dec_f = io.open(dec_data.file_path,'r')
    
      while true do

         -- Read in Batch Size
         iter = iter + 1
         model.output = {}
         local enc_line = {}
         local dec_line = {}
         while #enc_line < opts.batch_size do
            local tmp = enc_f:read("*l")
            table.insert(enc_line,tmp)
            tmp = dec_f:read("*l")
            table.insert(dec_line,tmp)
            if (tmp == nil) then break end
         end

         if #enc_line == 0 then break end

         -- Initialize Matrices
         enc_x, enc_y, dec_x, dec_y, batch = {}, {}, {}, {}, {}
         initializeBatch(#enc_line)
         initializeEncMat()
         initializeDecMat()

         collectgarbage()

         -- Load Matrices
         batch.enc_len_max = 0
         batch.dec_len_max = 0
         batch.dec_line_length = {}
         for i=1,#enc_line do
            if enc_line[i] ~= nil then
               enc_num_word, dec_num_word = loadMat(enc_line[i],dec_line[i],i)
               if enc_num_word > batch.enc_len_max then
                  batch.enc_len_max = enc_num_word
               end
               if dec_num_word > batch.dec_len_max then
                  batch.dec_len_max = dec_num_word
               end
               batch.dec_line_length[i] = dec_num_word
            end
         end

         -- Forward and Backward Prop
         -- print(enc_line[1])
         fp(enc_x,enc_y,dec_x,dec_y,batch)
         bp(enc_x,enc_y,dec_x,dec_y,batch)
         log(epoch, iter, max_iter)
         decode(epoch,iter,batch,dec_line)
         if batch.size ~= opts.batch_size then
            break
         end

         --os.execute("nvidia-smi")
      end

      enc_f:close()
      dec_f:close()

      -- Saving
      torch.save(opts.save_dir .. '/model.th7', 
                 {params, epoch + opts.start, opts.lr, stats})
   end         
end

run()   



