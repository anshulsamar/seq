require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'paths'
require 'math'
require 'utils/base'
require 'data'

local function bp_decoder(x, y, batch)
   decoder.dx:zero()
   for i = batch.dec_len_max, 1, -1 do
      local s = decoder.s[i-1]
      local derr = g_transfer_data(torch.ones(1))
      local input = {x[i], y[i], s}
      local output = {derr, decoder.ds}
      local tmp = decoder.net[i]:backward(input, output)[3]
      g_replace_table(decoder.ds, tmp)
      cutorch.synchronize()
   end

   decoder.norm = decoder.dx:norm()
   if decoder.dx:norm() > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/decoder.dx:norm()
      decoder.dx:mul(shrink_factor)
   end
   
   decoder.x:add(decoder.dx:mul(-opts.lr))
end

local function bp_mlp()
   for d = 1, 2 * opts.layers do
      local x, dx = mlp.mu.net[d]:getParameters()
      dx:zero()
      local x, dx = mlp.lsigs.net[d]:getParameters()
      dx:zero()

      local muNLL = torch.ones(opts.batch_size,opts.dec_rnn_size):cuda()
      muNLL = torch.cmul(decoder.ds[d], muNLL)
      local muKL = mlp.mu.s[d] * -2
      local lsigsNLL = torch.cmul(torch.cmul(decoder.ds[d],mlp.eps[d]),
                                torch.exp(mlp.lsigs.s[d] * 1/2)) * 1/2
      local p1 = torch.exp(mlp.lsigs.s[d])*(-1) 
      local p2 = torch.ones(opts.batch_size,opts.dec_rnn_size):cuda()
      local lsigsKL = p1 + p2
      mlp.mu.ds[d] = mlp.mu.net[d]:backward(input, muNLL - muKL)
      mlp.lsigs.ds[d] = mlp.lsigs.net[d]:backward(input, lsigsNLL - lsigsKL)
   end

   for d = 1, 2*opts.layers do
      local x, dx = mlp.mu.net[d]:getParameters()
      mlp.mu.norm = mlp.mu.norm + dx:norm()
      if dx:norm() > opts.max_grad_norm then
         local shrink_factor = opts.max_grad_norm/dx:norm()
         dx:mul(shrink_factor)
      end
      x:add(dx:mul(-opts.lr))

      local x, dx = mlp.lsigs.net[d]:getParameters()
      mlp.lsigs.norm = mlp.mu.norm + dx:norm()
      if dx:norm() > opts.max_grad_norm then
         local shrink_factor = opts.max_grad_norm/dx:norm()
         dx:mul(shrink_factor)
      end
      x:add(dx:mul(-opts.lr))
   end

   mlp.mu.norm = mlp.mu.norm/(2*opts.layers)
   mlp.lsigs.norm = mlp.lsigs.norm/(2*opts.layers)
end

local function bp_encoder(x, y, batch)
   encoder.dx:zero()
   for i = batch.enc_len_max, 1, -1 do
      for k = 1, batch.size do
         if batch.enc_line_length[k] == i then
            for d = 1, 2 * opts.layers do
               if opts.sgvb then
                  encoder.ds[d][k] = mlp.mu.ds[d][k] + mlp.lsigs.ds[d][k]
               else
                  encoder.ds[d][k] = decoder.ds[d][k]
               end
            end
         end
      end
      for k = batch.size + 1, opts.batch_size do
         for d = 1, 2 * opts.layers do
            encoder.ds[d][k]:zero()
         end
      end

      local s = encoder.s[i-1]
      local derr = g_transfer_data(torch.ones(1))
      local input = {x[i], y[i], s}
      local output = {derr, encoder.ds}
      local tmp = encoder.net[i]:backward(input, output)[3]
      g_replace_table(encoder.ds, tmp)
      cutorch.synchronize()
   end

   encoder.norm = encoder.dx:norm()

   if encoder.dx:norm() > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/encoder.dx:norm()
      encoder.dx:mul(shrink_factor)
   end

   encoder.x:add(encoder.dx:mul(-opts.lr))
end

function bp(enc_x,enc_y,dec_x,dec_y,batch,mode)
   if mode == 'test' then return end
   bp_decoder(dec_x, dec_y, batch)
   if opts.sgvb then bp_mlp() end
   bp_encoder(enc_x, enc_y, batch)
end
