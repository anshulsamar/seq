require 'torch'
require 'utils/base.lua'

local function fp_encoder(x, y, batch, mode)
   for i = 1, batch.enc_len_max do
      local s = encoder.s[i - 1]
      local ret = encoder.net[i]:forward({x[i], y[i], s})
      encoder.err[i] = ret[1]
      encoder.s[i] = ret[2]
   end
end

local function fp_decoder(x, y, batch, mode)
   for i = 1, batch.dec_len_max do
      local s = decoder.s[i - 1]
      local ret = decoder.net[i]:forward({x[i], y[i], s})
      -- during testime, feed output as input
      if mode == 'test' or mode == 'sweep' then
         local _, ind = torch.max(decoder.out[i],2)
         x[i + 1] = ind:select(2,1)
      end
      decoder.err[i] = ret[1]
      decoder.s[i] = ret[2]
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
      if mode == 'sweep' then
         for k = 1, batch.size do
            z[k] = mu + torch.cmul(sigma, -1 + (batch.size - k)*.02)
         end
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

