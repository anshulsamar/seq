-- Copyright (c) 2015 Anshul Samar
-- Copyright (c) 2014, Facebook, Inc. All rights reserved.
-- Licensed under the Apache License, Version 2.0 found in main folder
-- See original LSTM/LM code: github.com/wojzaremba/lstm

require 'io'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'paths'
require 'math'
require 'criterion/EncCriterion'
require 'criterion/DecCriterion'
require 'utils/base'
require 'dataLoader'


-- Global Data Structures
-- ---------------------
-- model: encoder, decoder, deltas, states, errors, etc. 
-- params: 'x' refers to weights/biases and 'dx' to gradient
-- data: lookup tables, indexes, etc.
-- opts: command line options
-- dec_output: softmax output of decoder (must be global)

local model = {}
local params = {encoderx ={}, encoderdx = {}, decoderx = {}, decoderdx = {}}
local data = {}
local opts
dec_output = {}

function transfer_data(x)
   return x:cuda()
end

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
   return transfer_data(module)

end

local function setupEncoder()
   local encoder = create_network(EncCriterion,enc_data.lookup,
                            enc_data.lookup_size,enc_data.vocab_size,
                            opts.enc_in_size, opts.enc_rnn_size)
   params.encoderx, params.encoderdx = encoder:getParameters()
   model.encoder = g_cloneManyTimes(encoder, enc_data.len_max)
   model.enc_s = {}
   model.enc_ds = {}
   for j = 0, enc_data.len_max do
      model.enc_s[j] = {}
      for d = 1, 2 * opts.layers do
         local outputStates = torch.zeros(opts.batch_size,opts.enc_rnn_size)
         model.enc_s[j][d] = transfer_data(outputStates)
      end
   end
   for d = 1, 2 * opts.layers do
      local deltas = torch.zeros(opts.batch_size,opts.enc_rnn_size)
      model.enc_ds[d] = transfer_data(deltas)
   end
   model.enc_norm_dw = 0
   model.enc_err = transfer_data(torch.zeros(enc_data.len_max))
end

local function setupDecoder()
   local decoder = create_network(DecCriterion,dec_data.lookup,
                            dec_data.lookup_size,dec_data.vocab_size,
                            opts.dec_in_size, opts.dec_rnn_size)
   params.decoderx, params.decoderdx = decoder:getParameters()
   model.decoder = g_cloneManyTimes(decoder, dec_data.len_max)
   model.dec_s = {}
   model.dec_ds = {}
   for j = 0, dec_data.len_max do
      model.dec_s[j] = {}
      for d = 1, 2 * opts.layers do
         local outputStates = torch.zeros(opts.batch_size,opts.dec_rnn_size)
         model.dec_s[j][d] = transfer_data(outputStates)
      end
   end
   for d = 1, 2 * opts.layers do
      local deltas = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      model.dec_ds[d] = transfer_data(deltas)
   end
   model.dec_norm_dw = 0
   model.dec_err = transfer_data(torch.zeros(dec_data.len_max))
end

local function fpSeq(x, y, batch_len_max, line_length, num_examples, state, err, net, test, system)

   if test then
      local x_init = torch.ones(opts.batch_size) * dec_data.default_index
      dec_x = {transfer_data(x_init)}
   end

   local ret
   for i = 1, batch_len_max do
      local s = state[i - 1]
      ret = net[i]:forward({x[i], y[i], s})
      if test and system == 'decoder' then -- and i < batch_len_max then
         local _, ind = torch.max(dec_output[i],2)
         table.insert(x,ind:select(2,1))
      end

      err[i] = ret[1]
      state[i] = ret[2]
      for j = 1, num_examples do
         if i > line_length[j]  then
            for d = 1, 2 * opts.layers do
               state[i][d][j]:zero()
            end
         end
      end

      for j = num_examples + 1, opts.batch_size do
         for d = 1, 2 * opts.layers do
            state[i][d][j]:zero()
         end
      end

   end


end

local function fp(enc_x, enc_y, dec_x, dec_y, batch, test)
   g_reset_s(model.enc_s,enc_data.len_max,opts)
   g_reset_s(model.dec_s,dec_data.len_max,opts)


   fpSeq(enc_x,enc_y,batch.enc_len_max, batch.enc_line_length, batch.size, 
         model.enc_s, model.enc_err, model.encoder, test, 'encoder')

   for j = 1, batch.size do
      for d = 1, 2 * opts.layers do
         model.dec_s[0][d][j] = model.enc_s[batch.enc_line_length[j]][d][j]
      end
   end

   fpSeq(dec_x, dec_y,batch.dec_len_max, batch.dec_line_length, batch.size, 
      model.dec_s, model.dec_err, model.decoder, test, 'decoder')
end


local function bpSeq(x, y, batch_len_max, num_examples, line_length, state, ds, net, grad, system)

   grad:zero()
   for i = batch_len_max, 1, -1 do

      if system == 'encoder' then
         for j = 1, num_examples do
            if line_length[j] == i then
               for d = 1, 2 * opts.layers do
                  ds[d][j] = model.dec_ds[d][j]
               end
            end
         end
         for j = num_examples + 1, opts.batch_size do
            for d = 1, 2 * opts.layers do
               ds[d][j]:zero()
            end
         end
      end

      local s = state[i-1]
      local derr = transfer_data(torch.ones(1))
      local input = {x[i], y[i], s}
      local output = {derr, ds}
      local tmp = net[i]:backward(input, output)[3]
      g_replace_table(ds, tmp)
      cutorch.synchronize()
   end

   if grad:norm() > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/grad:norm()
      grad:mul(shrink_factor)
   end

   return grad

end

local function bp(enc_x,enc_y,dec_x,dec_y,batch,test)

   if test then return end
   g_reset_ds(model.dec_ds,opts)   
   g_reset_ds(model.enc_ds,opts)   
   grad = bpSeq(dec_x, dec_y, 
                batch.dec_len_max, batch.size, batch.dec_line_length, 
                model.dec_s, model.dec_ds, model.decoder, 
                params.decoderdx, 'decoder')
   model.dec_norm_dw = grad:norm()
   params.decoderx:add(grad:mul(-opts.lr))

   grad = bpSeq(enc_x, enc_y, 
                batch.enc_len_max, batch.size, batch.enc_line_length, 
                model.enc_s, model.enc_ds, model.encoder, 
                params.encoderdx, 'encoder')
   model.enc_norm_dw = grad:norm()
   params.encoderx:add(grad:mul(-opts.lr))

end


local function getError(batch)
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


local function log(epoch, iter, test, stats, batch)
   local st 
   if test then 
      st = stats.test 
   else 
      st = stats.train 
   end

   st.enc_err, st.dec_err = getError(batch)
   st.avg_enc_err = ((st.avg_enc_err * (iter-1)) + st.enc_err)/iter
   st.avg_dec_err = ((st.avg_dec_err * (iter-1)) + st.dec_err)/iter
   local runtime = "train"
   if test then runtime = "test" end

   print(runtime .. ': epoch=' .. string.format('%02d',epoch + opts.start) ..  
         ', iter=' .. string.format('%03d',iter) ..
         ', enc_err=' .. string.format('%.2f',st.enc_err) ..
         ', avg_enc_err=' .. string.format('%.2f',st.avg_enc_err) ..
         ', dec_err=' .. string.format('%.2f',st.dec_err) .. 
         ', avg_dec_err=' .. string.format('%.2f',st.avg_dec_err) ..
         ', encdxNorm=' .. string.format('%.4f',model.enc_norm_dw) ..
         ', decdxNorm=' .. string.format('%.4f',model.dec_norm_dw) .. 
         ', lr=' ..  string.format('%.3f',opts.lr))

   st.avg_dec_err_epoch[epoch] = st.avg_dec_err

end


local function loadModel()
   filen = opts.run_dir .. '/model.th7'
   if (paths.filep(filen)) then
      print("Loading previous parameters")
      local oldModel = torch.load(opts.run_dir .. '/model.th7')
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

local function loadMat(line, index, i, x, y, dec)
   local indexes = {}
   local num_word = 0
   local last_word = ""
   local len = 0
   for _,word in ipairs(stringx.split(line,' ')) do
      if word ~= "" then
         len = len + 1
      end
   end

   for _,word in ipairs(stringx.split(line,' ')) do
      if word ~= "" then
         num_word = num_word + 1
         if index[word] == nil then
            word = '<unk>'
         end
         if dec then
            y[num_word][i] = index[word]
            indexes[word] = {index[word],word}
         else
            x[num_word][i] = index[word]
         end
         last_word = word
      end
   end

   if dec then
      for j=1,#indexes - 1 do
         x[j+1][i] = indexes[j][1]
      end
   end

   return num_word, last_word
end

local function load(enc_x, enc_y, enc_line, dec_x, dec_y, dec_line, batch)
   for i=1,#enc_line do
      if enc_line[i] ~= nil then
         enc_num_word, last_word = loadMat(enc_line[i],
                                           enc_data.index,i,
                                           enc_x,enc_y,false)
         dec_x[1][i] = enc_data.index[last_word]
         dec_num_word, _ = loadMat(dec_line[i],
                                   dec_data.index,i,
                                   dec_x,dec_y,true)
         if enc_num_word > batch.enc_len_max then
            batch.enc_len_max = enc_num_word
         end
         if dec_num_word > batch.dec_len_max then
            batch.dec_len_max = dec_num_word
         end
         batch.dec_line_length[i] = dec_num_word
         batch.enc_line_length[i] = enc_num_word
      end
   end

end

local function decode(epoch,iter,batch,enc_line,dec_line,test,opts)

   local indexes = {}
   for i=1,#dec_output do
      local y, ind = torch.max(dec_output[i],2)
      table.insert(indexes,ind)
   end

   local runtime = "train"
   if test then runtime = "test" end

   if epoch + opts.start - 1 >= 1 then
      local oldDecodeName = 'decode_' .. runtime .. '_' .. (epoch + opts.start - 1) 
         .. '*'
      os.execute('rm -rf ' .. opts.decode_dir .. oldDecodeName)
   end

   local decodeName = 'decode_' .. runtime .. '_' .. (epoch + opts.start) 
      .. '_' .. iter .. '.txt'
  
   local f = io.open(opts.decode_dir .. decodeName,'a+')

   for i=1,#dec_line do
      local num_words = batch.dec_line_length[i]
      local sentence = ''
      for j=1,num_words do
         sentence = sentence .. dec_data.rev_index[indexes[j][i][1]] .. ' '
      end
      f:write(enc_line[i] .. ' | ' .. sentence,'\n')
      f:flush()
   end

   f:close()

end

local function getOpts()
   local cmd = torch.CmdLine()

   -- Network
   cmd:option('-layers',2)
   cmd:option('-enc_in_size',100)
   cmd:option('-enc_rnn_size',200)
   cmd:option('-dec_in_size',100)
   cmd:option('-dec_rnn_size',200)
   cmd:option('-load_model',false)

   -- Training
   cmd:option('-max_grad_norm',5)
   cmd:option('-max_epoch',15)
   cmd:option('-anneal',true)
   cmd:option('-anneal_after',10)
   cmd:option('-decay',2)
   cmd:option('-weight_init',.1)
   cmd:option('-lr',0.7)
   cmd:option('-freq_floor',6)
   cmd:option('-start',0)
   cmd:option('-test',true)
   cmd:option('-train',true)
   cmd:option('-gpu',1)
   cmd:option('-stats',false)

   -- Data
   cmd:option('-share',true) -- share data and lookup table b/w enc/dec
   cmd:option('-batch_size',96)
   cmd:option('-data_dir','/deep/group/speech/asamar/nlp/data/numbers/')
   cmd:option('-enc_train_file','enc_train.txt')
   cmd:option('-dec_train_file','dec_train.txt')
   cmd:option('-enc_test_file','enc_test.txt')
   cmd:option('-dec_test_file','dec_test.txt')
   cmd:option('-glove',false)
   cmd:option('-glove_file','/deep/group/speech/asamar/nlp/glove/pretrained/glove.840B.300d.txt')
   cmd:option('-run_dir','/deep/group/speech/asamar/nlp/seq/numbers/')

   local opts = cmd:parse(arg)
   opts.decode_dir = opts.run_dir .. '/decode/'
   opts.enc_train_file = opts.data_dir .. opts.enc_train_file
   opts.dec_train_file = opts.data_dir .. opts.dec_train_file
   opts.enc_test_file = opts.data_dir .. opts.enc_test_file
   opts.dec_test_file = opts.data_dir .. opts.dec_test_file
   return opts
end

function run()
   -- Options
   print("\27[31mStarting Experiment\n---------------")
   opts = getOpts()
   g_init_gpu({opts.gpu})
   g_make_run_dir(opts)
   print(opts)
   print("Saving Options")
   torch.save(paths.concat(opts.run_dir,'opts.th7'),opts)

   -- Plot Err
   if opts.stats then
      g_plot_err(opts.run_dir .. 'model.th7')
      return
   end

   -- Data
   print("Loading Data")
   enc_data, dec_data = dataLoader.get(opts)
   if opts.share then
      dec_data = enc_data
   end

   -- Network
   print("\27[31mCreating Network\n----------------")
   print("Setting up Encoder")
   setupEncoder()
   print("Setting up Decoder")
   setupDecoder()
   local stats = {}
   stats.train = {avg_dec_err_epoch = {}}
   stats.test = {avg_dec_err_epoch = {}}
   if opts.load_model then 
      print('Loading Model')
      loadModel() 
   end

   -- Training
   print("\27[31mTraining\n----------")
   local test_options = {}
   if opts.train then
      table.insert(test_options, false)
   end

   if opts.test then
      table.insert(test_options, true)
   end

   for epoch=1,(opts.max_epoch - opts.start) do
      for _,test in ipairs(test_options) do
         -- Setup
         local iter = 0
         g_reset_stats(stats)

         -- Anneal
         if opts.anneal and (epoch + opts.start) > opts.anneal_after then
            opts.lr = opts.lr / opts.decay
         end

         -- Open Data
         local enc_f
         local dec_f

         if test then
            enc_f = io.open(enc_data.test_file,'r')
            dec_f = io.open(dec_data.test_file,'r')
         else
            enc_f = io.open(enc_data.train_file,'r')
            dec_f = io.open(dec_data.train_file,'r')
         end
         
         --local max_iter = math.ceil(enc_data.total_lines/opts.batch_size)
         
         while true do

            -- Read in Data
            iter = iter + 1
            dec_output = {}
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

            -- Initialize Matrices and Batch Struct
            local batch = g_initialize_batch(#enc_line)
            local enc_x, enc_y = g_initialize_mat(enc_data.len_max,
                                     enc_data.default_index, opts, batch)
            local dec_x, dec_y = g_initialize_mat(dec_data.len_max,
                                     dec_data.default_index, opts, batch)       

            collectgarbage()

            -- Load Matrices and Line Lengths
            load(enc_x, enc_y, enc_line, dec_x, dec_y, dec_line, batch)

            -- Forward and Backward Prop
            fp(enc_x,enc_y,dec_x,dec_y,batch,test)
            bp(enc_x,enc_y,dec_x,dec_y,batch,test)

            -- Log and Decode
            log(epoch,iter,test,stats,batch)
            decode(epoch,iter,batch,enc_line,dec_line,test,opts)

            if batch.size ~= opts.batch_size then
               break
            end
         end

         enc_f:close()
         dec_f:close()

         -- Save Model
         torch.save(opts.run_dir .. '/model.th7', 
                    {params, epoch + opts.start, opts.lr, stats})
      end         
   end
end

run()   



