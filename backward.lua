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

local function bp_decoder(x, y, batch)
   local x, dx = decoder.net[1]:getParameters()
   dx:zero()

   for i = batch.dec_len_max, 1, -1 do
      local s = decoder.s[i-1]
      local derr = g_transfer_data(torch.ones(1))
      local input = {x[i], y[i], s}
      local output = {derr, ds}
      local tmp = decoder.net[i]:backward(input, output)[3]
      g_replace_table(decoder.ds, tmp)
      cutorch.synchronize()
   end

   if dx:norm() > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/dx:norm()
      dx:mul(shrink_factor)
   end

   x:add(dx:mul(-opts.lr))

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
      local lsigsKL = torch.exp(mlp.lsigs.s[d])*(-1) + torch.ones(opts.batch_size,opts.dec_rnn_size):cuda()
      mlp.mu.ds[d] = model.mu.net[d]:backward(input, muNLL - muKL)
      mlp.lsigs.ds[d] = model.lsigs.net[d]:backward(input, lssNLL - lssKL)
   end

   for d = 1, 2 * opts.layers do
      local x, dx = model.lsm[d]:getParameters()
      x:add(dx:mul(-opts.lr))
      local x, dx = model.lss[d]:getParameters()
      x:add(dx:mul(-opts.lr))
   end


   for d = 1, 2 * opts.layers do
      local x, dx = mlp.mu.net[d]:getParameters()
      dx:zero()
      if dx:norm() > opts.max_grad_norm then
         local shrink_factor = opts.max_grad_norm/dx:norm()
         dx:mul(shrink_factor)
      end
      x:add(dx:mul(-opts.lr))

      local x, dx = mlp.lsigs.net[d]:getParameters()
      dx:zero()
      if dx:norm() > opts.max_grad_norm then
         local shrink_factor = opts.max_grad_norm/dx:norm()
         dx:mul(shrink_factor)
      end
      x:add(dx:mul(-opts.lr))
   end
end

local function bp_encoder(x, y, batch)
   local x, dx = encoder.net[1]:getParameters()
   dx:zero()

   for i = batch_len_max, 1, -1 do
      local s = state[i-1]
      local derr = g_transfer_data(torch.ones(1))
      local input = {x[i], y[i], s}
      local output = {derr, ds}
      local tmp = encoder.net[i]:backward(input, output)[3]
      g_replace_table(encoder.ds, tmp)
      cutorch.synchronize()
   end

   if dx:norm() > opts.max_grad_norm then
      local shrink_factor = opts.max_grad_norm/dx:norm()
      dx:mul(shrink_factor)
   end

   x:add(dx:mul(-opts.lr))
end

local function transfer_state(batch)

   for i = batch.enc_len_max, 1, -1 do
      for k = 1, batch.size do
         if batch.enc_line_length[k] == i then
            for d = 1, 2 * opts.layers do
               if opts.sgvb then
                  encoder.ds[d][k] = mlp.lsmds[d][k] + mlp.lssds[d][k]
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
   end

end

local function bp(enc_x,enc_y,dec_x,dec_y,batch,test)
   if test then return end
   g_reset_ds(decoder.ds,opts)   
   g_reset_ds(encoder.ds,opts)   
   bp_decoder(dec_x, dec_y, batch)
   if opts.sgvb then
      g_reset_ds(mlp.mu.ds, opts)
      g_reset_ds(mlp.lsigs.ds, opts)
      bp_mlp()
   end
   transfer_state(batch)
   bp_encoder(enc_x, enc_y, batch)
end
