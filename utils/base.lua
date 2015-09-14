--  Copyright (c) 2015 Anshul Samar
--  Copyright (c) 2014, Facebook, Inc. All rights reserved.
--  Licensed under the Apache License, Version 2.0 found in main folder
--  See original LSTM/LM code: github.com/wojzaremba/lstm
--  g_print method from Benjamin Marechal

require 'gnuplot'

function g_transfer_data(x)
   return x:cuda()
end

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

function g_init_gpu(gpu)
  print(string.format("Using %s-th gpu", gpu))
  cutorch.setDevice(gpu)
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
      local oldModel = torch.load(model_file)
      stats = oldModel[6]
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
   batch.enc_len_max = 0
   batch.enc_line_length = {}
   batch.enc_lengths = torch.zeros(size)
   batch.dec_len_max = 0
   batch.dec_line_length = {}
   batch.dec_lengths = torch.zeros(size)
   return batch
end

function g_initialize_mat(len_max, default_index, opts)
   local x = {}
   local y = {}

   for i=1,len_max do
      local x_init = torch.ones(opts.batch_size) * default_index
      table.insert(x,g_transfer_data(x_init))
      table.insert(y,g_transfer_data(torch.zeros(opts.batch_size)))
   end
   return x, y
   
end

function g_initialize_eps()
   for d = 1, 2 * opts.layers do
      if opts.sgvb then
         model.eps[d] = g_transfer_data(torch.zeros(opts.batch_size))
      end
   end
end

function g_make_run_dir(opts)
   if paths.dir(opts.run_dir) == nil then
      paths.mkdir(opts.run_dir)
      paths.mkdir(opts.decode_dir)
   end
end

function g_reset_encoder()
   for j = 0, enc_data.len_max do
      for d = 1, 2 * opts.layers do
         encoder.s[j][d]:zero()
      end
   end
   for d = 1, 2 * opts.layers do
      encoder.ds[d]:zero()
      encoder.out[d]:zero()
   end
end

function g_reset_decoder()
   for j = 0, dec_data.len_max do
      for d = 1, 2 * opts.layers do
         decoder.s[j][d]:zero()
      end
   end
   for d = 1, 2 * opts.layers do
      decoder.ds[d]:zero()
   end
   decoder.out = {}
end

function g_reset_mlp()
   for d = 1, 2 * opts.layers do
      mlp.lsigs.s[d]:zero()
      mlp.lsigs.ds[d]:zero()
      mlp.mu.s[d]:zero()
      mlp.mu.ds[d]:zero()
      local x, dx = mlp.mu.net[d]:getParameters()
      dx:zero()
      local x, dx = mlp.lsigs.net[d]:getParameters()
      dx:zero()
   end
   mlp.mu.norm = 0
   mlp.lsigs.norm = 0
end

function g_reset_stats(stats)
   stats.train.avg_dec_err = 0
   stats.train.dec_err = 0
   stats.test.avg_dec_err = 0
   stats.test.dec_err = 0
end

function g_print_mod(mlp)
   for indexNode, node in ipairs(mlp.forwardnodes) do
      if node.data.module then
         print(node.data.module)
      end
   end
end

function g_print(text,color)
   if color == 'red' then
      print('\27[31m' .. text)
   elseif color == 'green' then
      print('\27[32m' .. text)
   elseif color == 'yellow' then
      print('\27[33m' .. text)
   elseif color == 'blue' then
      print('\27[34m' .. text)
   elseif color == 'purple' then
      print('\27[35m' .. text)
   elseif color == 'cyan' then
      print('\27[36m' .. text)
   else
      print(text)
   end
end



