
local function getError(batch)
   local tot_enc_err = 0

   for i = 1, batch.enc_len_max do
      tot_enc_err = tot_enc_err + encoder.err[i]
   end
   tot_enc_err = tot_enc_err * opts.batch_size / batch.size

   local tot_dec_err = 0
   for i = 1, batch.dec_len_max do
      tot_dec_err = tot_dec_err + decoder.err[i]
   end
   tot_dec_err = tot_dec_err * opts.batch_size / batch.size
   return tot_enc_err, tot_dec_err
end


function log(epoch, iter, mode, stats, batch)
   local st 
   if mode == 'test' then 
      st = stats.test 
   else 
      st = stats.train 
   end

   st.enc_err, st.dec_err = getError(batch)
   st.avg_dec_err = ((st.avg_dec_err * (iter-1)) + st.dec_err)/iter

   print(mode .. ': epoch=' .. string.format('%02d',epoch + opts.start) ..  
         ', iter=' .. string.format('%03d',iter) ..
         -- ', enc_err=' .. string.format('%.2f',st.enc_err) ..
         -- ', avg_enc_err=' .. string.format('%.2f',st.avg_enc_err) ..
         ', dec_err=' .. string.format('%.2f',st.dec_err) .. 
         ', avg_dec_err=' .. string.format('%.2f',st.avg_dec_err) ..
         ', encdxNorm=' .. string.format('%.4f',encoder.norm) ..
         ', decdxNorm=' .. string.format('%.4f',decoder.norm) .. 
         ', mudxNorm=' .. string.format('%.4f',mlp.mu.norm) .. 
         ', lsigsdxNorm=' .. string.format('%.4f',mlp.lsigs.norm) .. 
         ', lr=' ..  string.format('%.3f',opts.lr))

   st.avg_dec_err_epoch[epoch] = st.avg_dec_err
end

function decode(epoch,iter,batch,enc_line,dec_line,mode,opts)
   local indexes = {}
   for i=1,#decoder.out do
      local y, ind = torch.max(decoder.out[i],2)
      table.insert(indexes,ind)
   end

   if epoch + opts.start - 1 >= 1 then
      local old = mode .. '_' .. (epoch + opts.start - 1) 
         .. '*'
      os.execute('rm -rf ' .. opts.decode_dir .. old)
   end

   local decode_f = mode .. '_' .. (epoch+opts.start) .. '_' .. iter .. '.txt'
   local f = io.open(opts.decode_dir .. decode_f, 'a+')

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
