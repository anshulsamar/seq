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
require 'utils/base'
require 'utils/process'
require 'utils/load'
require 'data'
require 'forward'
require 'backward'
require 'network'

-- Global Data Structures

encoder = {net = {}, s = {}, ds = {}, out = {}, err = {}, norm = 0, x, dx}
decoder = {net = {}, s = {}, ds = {}, out = {}, err = {}, norm = 0, x, dx}
mlp = {lsigs = {}, mu = {}, eps = {}}
mlp.lsigs = {net = {}, s = {}, ds = {}, err = {}, norm = 0}
mlp.mu = {net = {}, s = {}, ds = {}, err = {}, norm = 0}
enc_data = {}
dec_data = {}
opts = {}

local function get_opts()
   local cmd = torch.CmdLine()

   -- Network
   torch.manualSeed(1)
   cmd:option('-layers',2)
   cmd:option('-enc_in_size',100)
   cmd:option('-enc_rnn_size',200)
   cmd:option('-dec_in_size',100)
   cmd:option('-dec_rnn_size',200)
   cmd:option('-load_model',false)

   -- Training
   cmd:option('-sgvb',false)
   cmd:option('-max_grad_norm',5)
   cmd:option('-anneal',true)
   cmd:option('-anneal_after',15)
   cmd:option('-decay',2)
   cmd:option('-weight_init',.1)
   cmd:option('-lr',0.7)
   cmd:option('-start',0)
   cmd:option('-max_epoch',20)
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
   cmd:option('-glove_file','/deep/group/speech/asamar/nlp/' ..
                 'glove/pretrained/glove.840B.300d.txt')
   cmd:option('-run_dir','/deep/group/speech/asamar/nlp/seq/numbers/')

   -- Load Options, Make Paths
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
   g_print('Starting Experiment\n---------------','red')
   opts = get_opts()
   g_init_gpu(opts.gpu)
   g_make_run_dir(opts)
   print(opts)
   print("Saving Options")
   torch.save(paths.concat(opts.run_dir,'opts.th7'),opts)

   -- Data
   print("Loading Data")
   enc_data, dec_data = data.get(opts)
   if opts.share then
      dec_data = enc_data
   end

   -- Network
   g_print("Creating Network\n----------------",'red')
   print("Setting up Encoder")
   setup_encoder()
   print("Setting up Decoder")
   setup_decoder()

   if opts.sgvb then setup_mlp() end

   -- Stats
   if opts.stats then g_plot_err(opts.run_dir .. 'model.th7') return end
   local stats = {}
   stats.train = {avg_dec_err_epoch = {}}
   stats.test = {avg_dec_err_epoch = {}}
   
   -- Training
   g_print("Training\n----------", 'red')
   local modes = {}
   if opts.train then table.insert(modes, 'train') end
   if opts.test then table.insert(modes, 'test') end

   for epoch=1,(opts.max_epoch - opts.start) do
      for _,mode in ipairs(modes) do
         -- Setup
         local iter = 0

         g_reset_stats(stats)

         -- Anneal
         if opts.anneal and (epoch + opts.start) > opts.anneal_after then
            opts.lr = opts.lr / opts.decay
         end

         -- Open Data
         local enc_f, dec_f

         if mode == 'test' then
            enc_f = io.open(enc_data.test_file,'r')
            dec_f = io.open(dec_data.test_file,'r')
         else
            enc_f = io.open(enc_data.train_file,'r')
            dec_f = io.open(dec_data.train_file,'r')
         end
         
         while true do
            -- Read in Data
            iter = iter + 1
            decoder.out = {}
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
            g_reset_encoder()
            g_reset_decoder()
            if opts.sgvb then g_reset_mlp() end
            fp(enc_x,enc_y,dec_x,dec_y,batch,mode)
            bp(enc_x,enc_y,dec_x,dec_y,batch,mode)

            -- Log and Decode
            log(epoch,iter,mode,stats,batch)
            decode(epoch,iter,batch,enc_line,dec_line,mode,opts)

            if batch.size ~= opts.batch_size then
               break
            end
         end

         enc_f:close()
         dec_f:close()

         -- Save Model
         
         local save_struct = {{}, epoch + opts.start, opts.lr, stats}
         torch.save(opts.run_dir .. '/model.th7', save_struct)
                    
      end         
   end
end

run()   



