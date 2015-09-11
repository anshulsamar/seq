
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

function decode(epoch,iter,batch,enc_line,dec_line,test,opts)

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
