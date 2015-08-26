require 'io'
require 'torch'
require 'reload'
require 'paths'

local modname = ...
local M = {}
_G[modname] = M
package.loaded[modname] = M

function M.getGlove(path,index,size)

   local word_emb = {}
   local count = 0
   local gloveLines = 0
   for line in io.lines(path) do
      print(line)
      local split = stringx.split(line,' ')
      local key = table.remove(split,1)
      if (index[key] ~= nil) then
         local vec = {}
         for _,val in ipairs(split) do
            local num = tonumber(val)
            if (num ~= nil) then
               table.insert(vec,tonumber(val))
            end
         end
         word_emb[key] = torch.Tensor(vec)
         count = count + 1
         if (count % 1000 == 0) then
            io.write(count/1000 .. ', ')
            io.flush()
         end
      end
      gloveLines = gloveLines + 1
      if (count == size) then
         break
      end
   end

   io.write('done.','\n')
   return word_emb
end

function M.parseGut(data_path,file_path)

   local index = {}
   local word_count = {}
   local rev_index = {}
   local vocab_size = 0
   local len_max = 0
   local f = io.open(file_path,'w')
   local total_lines = 0
   for filename in io.popen('ls -a ' .. data_path):lines() do
      if filename ~= '.' and filename ~= '..' and filename:sub(#filename,#filename) ~= '~' and filename:sub(#filename,#filename) ~= '#' then
         print("Loading " .. filename)
         local data = io.open(data_path .. filename, 'r')
         for line in data:read('*a'):gmatch('.-[%.%?!]') do
            local len = 0
            for word in line:lower():gmatch("%a+") do
               len = len + 1
            end
            if (len < 25 and len > 0) then 
               total_lines = total_lines + 1
               for word in line:lower():gmatch("%a+") do
                  f:write(word .. ' ')
                  if (index[word] == nil) then
                     vocab_size = vocab_size + 1
                     index[word] = vocab_size
                     word_count[word] = 1
                     rev_index[vocab_size] = word
                  else
                     word_count[word] = word_count[word] + 1
                  end

               end
               if len > 0 then
                  f:write('<EOS> \n')
               end
               if len > len_max then
                  len_max = len
               end
            end
         end
         data:close()
      end
   end

   if index['<EOS>'] == nil then
      vocab_size = vocab_size + 1
      index['<EOS>'] = vocab_size
      rev_index[vocab_size] = '<EOS>'
   end
   len_max = len_max + 1 --account for EOM
   f:close()
   return index, rev_index, word_count, vocab_size, len_max, total_lines

end

function M.parseMT(data_path,file_path,opts)

   local index = {}
   local word_count = {}
   local rev_index = {}
   local vocab_size = 0
   local len_max = 0
   local f = io.open(file_path,'w')
   local total_lines = 0
   for filename in io.popen('ls -a ' .. data_path):lines() do
      if filename ~= '.' and filename ~= '..' and filename:sub(#filename,#filename) ~= '~' and filename:sub(#filename,#filename) ~= '#' then
         print("Loading  " .. filename)
         local data = io.open(data_path .. filename, 'r')
         while true do
            local line = data:read()
            if line == nil then break end
            local words = stringx.split(line:lower())
            local len = words:len()
            if (len > 0) then 
               total_lines = total_lines + 1
               for _,word in ipairs(words) do
                  f:write(word .. ' ')
                  if (index[word] == nil) then
                     vocab_size = vocab_size + 1
                     word_count[word] = 1
                     table.insert(rev_index, word)
                     index[word] = vocab_size
                  else
                     word_count[word] = word_count[word] + 1
                  end
               end
               if len > 0 then
                  f:write('<EOS> \n')
               end
               if len > len_max then
                  len_max = len
               end
               old_line = line
            end
         end
         data:close()
      end
   end

   print("Total Lines " .. total_lines)
   print("Original Vocab Size " .. vocab_size)

   vocab_size = 0
   rev_index2 = {}
   index = {}
   
   for key,val in ipairs(rev_index) do
      if word_count[val] >= opts.freq_floor and val ~= '<EOS>' and val ~= '<UNK>' then
         table.insert(rev_index2,val)
         vocab_size = vocab_size + 1
      end
   end

   print("High Freq Vocab Size " .. vocab_size)
   
   rev_index = rev_index2
   table.insert(rev_index,'<EOS>')
   table.insert(rev_index,'<UNK>')
   vocab_size = vocab_size + 2

   print("Vocab Size with Add Tokens " .. vocab_size)

   for key,val in ipairs(rev_index) do
      index[val] = key
   end

   len_max = len_max + 1 --account for EOS

   print("Len Max " .. len_max)

   f:close()
   return index, rev_index, word_count, vocab_size, len_max, total_lines
end

function M.parsePenn(data_path,file_path,opts)

   local index = {}
   local word_count = {}
   local rev_index = {}
   local vocab_size = 0
   local len_max = 0
   local f = io.open(file_path,'w')
   local total_lines = 0
   for filename in io.popen('ls -a ' .. data_path):lines() do
      if filename ~= '.' and filename ~= '..' and filename:sub(#filename,#filename) ~= '~' and filename:sub(#filename,#filename) ~= '#' then
         print("Loading  " .. filename)
         local data = io.open(data_path .. filename, 'r')
         while true do
            local line = data:read()
            if line == nil then break end
            local words = stringx.split(line:lower())
            local len = words:len()
            if (len > 0) then 
               total_lines = total_lines + 1
               for _,word in ipairs(words) do
                  f:write(word .. ' ')
                  if (index[word] == nil) then
                     vocab_size = vocab_size + 1
                     word_count[word] = 1
                     table.insert(rev_index, word)
                     index[word] = vocab_size
                  else
                     word_count[word] = word_count[word] + 1
                  end
               end
               if len > 0 then
                  f:write('<EOS>\n')
               end
               if len > len_max then
                  len_max = len
               end
               old_line = line
            end
         end
         data:close()
      end
   end

   print("Total Lines " .. total_lines)
   print("Original Vocab Size " .. vocab_size)

   table.insert(rev_index,'<EOS>')
   table.insert(rev_index,'<UNK>')
   vocab_size = vocab_size + 2

   print("Vocab Size with EOS and UNK Token" .. vocab_size)

   for key,val in ipairs(rev_index) do
      index[val] = key
   end

   len_max = len_max + 1 --account for EOS

   print("Len Max " .. len_max)

   f:close()
   return index, rev_index, word_count, vocab_size, len_max, total_lines
end

function M.load(d,opts)

   print("Getting Data")

   if paths.filep(d.saved_vocab_path) == false or paths.filep(d.file_path) == false then
      if opts.parser == 'Gut' then
         print("Using Gutenberg Parser")
         d.index, d.rev_index, d.word_count, d.vocab_size, d.len_max, d.total_lines = M.parseGut(d.data_path,d.file_path)
      end
      if opts.parser == 'MT' then
         print("Using MT Parser")
         d.index, d.rev_index, d.word_count, d.vocab_size, d.len_max, d.total_lines = M.parseMT(d.data_path,d.file_path,opts)
      end
      if opts.parser == 'penn' then
         print("Using Penn Parser")
         d.index, d.rev_index, d.word_count, d.vocab_size, d.len_max, d.total_lines = M.parsePenn(d.data_path,d.file_path,opts)       
      end
      print("Saving")
      torch.save(d.saved_vocab_path,{d.index,d.rev_index,d.word_count, d.vocab_size,d.len_max,d.total_lines})
   else
      print("Loading Saved Data")
      d.index, d.rev_index, d.word_count, d.vocab_size, d.len_max, d.total_lines = unpack(torch.load(d.saved_vocab_path))
   end

   if paths.filep(d.saved_word_path) == false then
      d.word_emb = {}
      if opts.glove then 
         print("Getting Glove")
         -- Subtracting 1 from vocab_size because it includes <EOM> and <UNK>
         d.word_emb = M.getGlove(d.raw_word_path,d.index,d.vocab_size-2)
      end
      print("Saving Embeddings")
      torch.save(d.saved_word_path,d.word_emb)
   else
      print("Loading Saved Embeddings")
      d.word_emb = torch.load(d.saved_word_path)
   end

   if paths.filep(d.saved_lookup_path) == false then

      print("Generating Lookup Table")

      d.default_index = d.vocab_size + 1 -- for the default lookup of zeros
      d.lookup_size = d.default_index
      d.lookup = torch.Tensor(d.lookup_size,d.dim)
      d.lookup[d.default_index] = torch.zeros(d.dim)
      d.lookup[d.vocab_size] = torch.randn(d.dim) --<UNK> symbol

      for word,num in pairs(d.index) do
         if (d.word_emb[word] ~= nil) then
            d.lookup[num] = d.word_emb[word]
         else
            d.lookup[num] = torch.randn(d.dim)
         end
      end

      print("Saving Lookup Table")
      torch.save(d.saved_lookup_path,{d.default_index, d.lookup_size, d.lookup})
   else
      print("Loading Lookup Table")
      d.default_index, d.lookup_size, d.lookup = unpack(torch.load(d.saved_lookup_path))
   end


end

function M.get(opts)

   print("\27[31mEncoder Data\n-------------")

   local enc_d = {}
   enc_d.base_path = opts.base_path
   enc_d.data_path =  opts.data_dir_from .. '/enc/'
   enc_d.file_path = opts.data_dir_to .. 'enc.txt'
   enc_d.raw_word_path = opts.glove_path
   enc_d.saved_word_path = opts.data_dir_to .. 'wordEnc.th7'
   enc_d.saved_vocab_path = opts.data_dir_to .. 'vocabEnc.th7'
   enc_d.saved_lookup_path = opts.data_dir_to .. 'lookupEnc.th7'
   enc_d.dim = opts.in_size
   enc_d.index = {}
   enc_d.rev_index = {}
   enc_d.word_emb = {}
   enc_d.lookup = {}
   enc_d.vocab_size = 0
   enc_d.len_max = 0
   enc_d.total_lines = 0
   enc_d.high_freq = 0
   if paths.filep(enc_d.file_path .. '.shuf') then 
      os.execute("rm " .. enc_d.file_path .. '.shuf')
   end
   M.load(enc_d,opts)

   print("\27[31mDecoder Data\n-------------")

   local dec_d = {}
   dec_d.base_path = opts.base_path
   dec_d.data_path =  opts.data_dir_from .. '/dec/'
   dec_d.file_path = opts.data_dir_to .. 'dec.txt'
   dec_d.raw_word_path = opts.glove_path
   dec_d.saved_word_path = opts.data_dir_to .. 'wordDec.th7'
   dec_d.saved_vocab_path = opts.data_dir_to .. 'vocabDec.th7'
   dec_d.saved_lookup_path = opts.data_dir_to .. 'lookupDec.th7'
   dec_d.dim = opts.in_size
   dec_d.index = {}
   dec_d.rev_index = {}
   dec_d.word_emb = {}
   dec_d.lookup = {}
   dec_d.vocab_size = 0
   dec_d.len_max = 0
   dec_d.total_lines = 0
   dec_d.high_freq = 0
   if paths.filep(dec_d.file_path .. '.shuf') then 
      os.execute("rm " .. dec_d.file_path .. '.shuf')
   end
   M.load(dec_d,opts)

   print("\27[31mPost Processing\n---------------")
   
   print("Shuffle Data")

   --os.execute("paste -d \':\' " .. enc_d.file_path .. ' ' .. dec_d.file_path .. " | shuf | awk -v FS=\":\" \'{ print $1 > \"" .. enc_d.file_path .. '.shuf' .. "\" ; print $2 > \"" .. dec_d.file_path .. '.shuf' .. "\" }\'")

   --enc_d.file_path = enc_d.file_path .. '.shuf'
   --dec_d.file_path = dec_d.file_path .. '.shuf'

   --cite author

   return enc_d, dec_d
end


