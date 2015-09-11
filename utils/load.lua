local function loadModel()
   filen = opts.run_dir .. '/model.th7'
   if (paths.filep(filen)) then
      print("Loading previous parameters")
      local oldModel = torch.load(opts.run_dir .. '/model.th7')
      params.encoderx:copy(oldModel[1].encoderx)
      params.encoderdx:copy(oldModel[1].encoderdx)
      params.decoderx:copy(oldModel[1].decoderx)
      params.decoderdx:copy(oldModel[1].decoderdx)
      opts.start = oldModel[2]
      opts.lr = oldModel[3]
      stats = oldModel[4]
   else
      print('No model to load, training from scratch')
   end
end


local function loadMat(line, index, i, x, y, dec)
   local indexes = {}
   local num_word = 0
   local last_word = ""
   local len = 0
   for _,word in ipairs(stringx.split(line,' ')) do
      if word ~= "" then
         len = len + 1
      end
   end

   for _,word in ipairs(stringx.split(line,' ')) do
      if word ~= "" then
         num_word = num_word + 1
         if index[word] == nil then
            word = '<unk>'
         end
         if dec then
            y[num_word][i] = index[word]
            indexes[word] = {index[word],word}
         else
            x[num_word][i] = index[word]
         end
         last_word = word
      end
   end

   if dec then
      for j=1,#indexes - 1 do
         x[j+1][i] = indexes[j][1]
      end
   end

   return num_word, last_word
end

local function load(enc_x, enc_y, enc_line, dec_x, dec_y, dec_line, batch)
   for i=1,#enc_line do
      if enc_line[i] ~= nil then
         enc_num_word, last_word = loadMat(enc_line[i],
                                           enc_data.index,i,
                                           enc_x,enc_y,false)
         dec_x[1][i] = enc_data.index[last_word]
         dec_num_word, _ = loadMat(dec_line[i],
                                   dec_data.index,i,
                                   dec_x,dec_y,true)
         if enc_num_word > batch.enc_len_max then
            batch.enc_len_max = enc_num_word
         end
         if dec_num_word > batch.dec_len_max then
            batch.dec_len_max = dec_num_word
         end
         batch.dec_line_length[i] = dec_num_word
         batch.enc_line_length[i] = enc_num_word
      end
   end

end
