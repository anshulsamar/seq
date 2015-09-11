require 'torch'
require 'utils/base.lua'

local function fp_encoder(x, y, batch, test)
   local ret
   for i = 1, batch.enc_len_max do
      local s = encoder.s[i - 1]
      ret = encoder.net[i]:forward({x[i], y[i], s})
      encoder.err[i] = ret[1]
      encoder.s[i] = ret[2]

      for j = 1, batch.size do
         if i > batch.enc_line_length[j]  then
            for d = 1, 2 * opts.layers do
               encoder.s[i][d][j]:zero()
            end
         end
      end

      for j = batch.size + 1, opts.batch_size do
         for d = 1, 2 * opts.layers do
            encoder.s[i][d][j]:zero()
         end
      end
   end
end

local function fp_decoder(x, y, batch, test)
   if test then
      local x_init = torch.ones(opts.batch_size) * dec_data.default_index
      x = {g_transfer_data(x_init)}
   end

   local ret
   for i = 1, batch.dec_len_max do
      local s = decoder.s[i - 1]
      ret = decoder.net[i]:forward({x[i], y[i], s})
      if test then
         local _, ind = torch.max(dec_output[i],2)
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

local function fp_mlp()
   for d = 1, 2 * opts.layers do
      model.eps[d] = torch.zeros(opts.batch_size,opts.dec_rnn_size)
      g_transfer_data(model.eps[d])
      for k = 1, batch.size do
         local init = torch.ones(opts.dec_rnn_size) * torch.normal(0,1)
         model.eps[d][k] = g_transfer_data(init)
      end
   end

   for k = 1, batch.size do
      for d = 1, 2 * opts.layers do
         encoder.out[d][k] = encoder.s[batch.enc_line_length[k]][d][k]
      end
   end

   for d = 1, 2 * opts.layers do
      local input = model.enc_out[d]
      local mu = mlp.mu.net[d]:forward(input)
      mlp.mu.s[d] = mu
      local lsigs = mlp.lsigs.net[d]:forward(input)
      mlp.lsigs.s[d] = lsigs
      local sigma = torch.exp(lsigs * 1/2)
      local z = mu + torch.cmul(sigma, mlp.eps[d])
      decoder.s[0][d] = z
   end
end

function fp(enc_x, enc_y, dec_x, dec_y, batch, test)

   g_reset_s(encoder.s,enc_data.len_max,opts)
   g_reset_s(decoder.s,dec_data.len_max,opts)
   g_reset_ds(model.enc_out,opts)

   fp_encoder(enc_x, enc_y, batch, test)

   if opts.sgvb then 
      fp_mlp()
   else
      for k = 1, batch.size do
         for d = 1, 2 * opts.layers do
            decoder.s[0][d][k] = encoder.s[batch.enc_line_length[k]][d][k]
         end
      end
   end

   fp_decoder(dec_x, dec_y, batch, test)
end
