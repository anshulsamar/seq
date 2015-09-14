require 'torch'

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

function load(enc_x, enc_y, enc_line, dec_x, dec_y, dec_line, batch)
   for i=1,#enc_line do
      if enc_line[i] ~= nil then
         enc_line[i] = string.gsub(enc_line[i],'<eom>','')
         batch.enc_line_length[i], last_word = loadMat(enc_line[i],
                                           enc_data.index,i,
                                           enc_x,enc_y,false)
         dec_line[i] = dec_line[i] .. ' <eos>'
         dec_x[1][i] = dec_data.index['<eos>']
         batch.dec_line_length[i], _ = loadMat(dec_line[i],
                                   dec_data.index,i,
                                   dec_x,dec_y,true)
         if batch.enc_line_length[i] > batch.enc_len_max then
            batch.enc_len_max = enc_num_word
         end
         if batch.dec_line_length[i] > batch.dec_len_max then
            batch.dec_len_max = dec_num_word
         end
      end
   end
end
