require 'torch'

local function load_mat(line, index, i, x, y, system)
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
         if system == 'decoder' then
            y[num_word][i] = index[word]
            indexes[word] = {index[word],word}
         else
            if opts.reverse then
               x[len - num_word + 1][i] = index[word]
            else
               x[num_word][i] = index[word]
            end
         end
         last_word = word
      end
   end

   if system == 'decoder' then
      for j=1,#indexes - 1 do
         x[j+1][i] = indexes[j][1]
      end
   end

   return num_word
end

function load(enc_x, enc_y, enc_line, dec_x, dec_y, dec_line, batch, mode)
   for i=1,#enc_line do
      if enc_line[i] ~= nil then
         batch.enc_line_length[i] = load_mat(enc_line[i],
                                             enc_data.index,i,
                                             enc_x,enc_y,'encoder')
         dec_x[1][i] = dec_data.index['<eos>']
         batch.dec_line_length[i] = load_mat(dec_line[i],
                                             dec_data.index,i,
                                             dec_x,dec_y,'decoder')
         if batch.enc_line_length[i] > batch.enc_len_max then
            batch.enc_len_max = batch.enc_line_length[i]
         end
         if batch.dec_line_length[i] > batch.dec_len_max then
            batch.dec_len_max = batch.dec_line_length[i]
         end
      end
   end
end
