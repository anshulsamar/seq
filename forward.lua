require 'torch'
require 'utils/base.lua'

local function fp_encoder(x, y, batch, mode)
   local ret
   for i = 1, batch.enc_len_max do
      local s = encoder.s[i - 1]
      ret = encoder.net[i]:forward({x[i], y[i], s})
      encoder.err[i] = ret[1]
      encoder.s[i] = ret[2]

      for k = 1, batch.size do
         if i > batch.enc_line_length[k]  then
            for d = 1, 2 * opts.layers do
               encoder.s[i][d][k]:zero()
            end
         end
      end

      for k = batch.size + 1, opts.batch_size do
         for d = 1, 2 * opts.layers do
            encoder.s[i][d][k]:zero()
         end
      end
   end
end

local function fp_decoder(x, y, batch, mode)
   if mode == 'test' then
      local x_init = torch.ones(opts.batch_size) * dec_data.default_index
      x = {g_transfer_data(x_init)}
   end

   local ret
   for i = 1, batch.dec_len_max do
      local s = decoder.s[i - 1]
      ret = decoder.net[i]:forward({x[i], y[i], s})
      if mode == 'test' then
         local _, ind = torch.max(decoder.out[i],2)
         table.insert(x,ind:select(2,1))
      end

      decoder.err[i] = ret[1]
      decoder.s[i] = ret[2]

      for j = 1, batch.size do
         if i > batch.dec_line_length[j]  then
            for d = 1, 2 * opts.layers do
               decoder.s[i][d][j]:zero()
            end
         end
      end

      for j = batch.size + 1, opts.batch_size do
         for d = 1, 2 * opts.layers do
            decoder.s[i][d][j]:zero()
         end
      end
   end
end

local function fp_mlp(batch)
   for d = 1, 2 * opts.layers do
      mlp.eps[d] = torch.zeros(opts.batch_size,opts.dec_rnn_size):cuda()
      for k = 1, batch.size do
         local init = torch.ones(opts.dec_rnn_size) * torch.normal(0,1)
         mlp.eps[d][k] = g_transfer_data(init)
      end
   end
   
   for k = 1, batch.size do
      for d = 1, 2 * opts.layers do
         encoder.out[d][k] = encoder.s[batch.enc_line_length[k]][d][k]
      end
   end

   for d = 1, 2 * opts.layers do
      local input = encoder.out[d]
      local mu = mlp.mu.net[d]:forward(input)
      mlp.mu.s[d] = mu
      local lsigs = mlp.lsigs.net[d]:forward(input)
      mlp.lsigs.s[d] = lsigs
      local sigma = torch.exp(lsigs * 1/2)
      local z = mu + torch.cmul(sigma, mlp.eps[d])
      for k = batch.size + 1, opts.batch_size do
         z[k]:zero()
      end
      decoder.s[0][d] = z
   end
end

local function transfer_s(batch)
   for k = 1, batch.size do
      for d = 1, 2 * opts.layers do
         decoder.s[0][d][k] = encoder.s[batch.enc_line_length[k]][d][k]
      end
   end
end

function fp(enc_x, enc_y, dec_x, dec_y, batch, mode)
   fp_encoder(enc_x, enc_y, batch, mode)
   if opts.sgvb then fp_mlp(batch) else transfer_s(batch) end
   fp_decoder(dec_x, dec_y, batch, mode)
end

