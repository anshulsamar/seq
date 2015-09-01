--  Copyright (c) 2015 Anshul Samar
--  Copyright (c) 2014, Facebook, Inc. All rights reserved.
--  Licensed under the Apache License, Version 2.0 found in main folder
--  See original LSTM/LM code: github.com/wojzaremba/lstm

require 'gnuplot'

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

function g_init_gpu(args)
  local gpuidx = args
  gpuidx = gpuidx[1] or 1
  print(string.format("Using %s-th gpu", gpuidx))
  cutorch.setDevice(gpuidx)
  g_make_deterministic(1)
end

function g_make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function g_plot_err(model_file)
   if stats == nil then
      local oldModel = torch.load(modelFile)
      stats = oldModel[4]
   end
   print('Train Avg Dec Err')
   print(stats.train.avg_dec_err_epoch)
   print('Test Avg Dec Err')
   print(stats.test.avg_dec_err_epoch)
   gnuplot.plot({torch.Tensor(stats.train.avg_dec_err_epoch)},
                {torch.Tensor(stats.test.avg_dec_err_epoch)})
   gnuplot.title('Average Decoder Error vs Epochs')
   gnuplot.xlabel('Epoch')
   gnuplot.ylabel('Negative Log Likelihood')
end

function g_initialize_batch(size)
   local batch = {}
   batch.size = size
   batch.enc_lengths = torch.zeros(size)
   batch.dec_lengths = torch.zeros(size)
   batch.enc_len_max = 0
   batch.dec_len_max = 0
   batch.dec_line_length = {}
   batch.enc_line_length = {}
   return batch
end

function g_initialize_mat(len_max, default_index, opts)
   local x = {}
   local y = {}

   for i=1,len_max do
      local x_init = torch.ones(opts.batch_size) * default_index
      table.insert(x,transfer_data(x_init))
      table.insert(y,transfer_data(torch.zeros(opts.batch_size)))
   end
   return x, y
end

function g_make_run_dir(opts)
   if paths.dir(opts.run_dir) == nil then
      paths.mkdir(opts.run_dir)
      paths.mkdir(opts.decode_dir)
   end
end

function g_reset_s(state, len_max, opts)
   for j = 0, len_max do
      for d = 1, 2 * opts.layers do
         state[j][d]:zero()
      end
   end
end

function g_reset_ds(ds, opts)
   for d = 1, 2 * opts.layers do
      ds[d]:zero()
   end
end

function g_reset_stats(stats)
   stats.train.avg_enc_err = 0
   stats.train.avg_dec_err = 0
   stats.train.dec_err = 0
   stats.train.enc_err = 0
   stats.test.avg_enc_err = 0
   stats.test.avg_dec_err = 0
   stats.test.dec_err = 0
   stats.test.enc_err = 0
end


