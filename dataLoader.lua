require 'io'
require 'torch'
require 'reload'

local modname = ...
local M = {}
_G[modname] = M
package.loaded[modname] = M

function M.getWordEmbeddings(path,index,size)

   local word_emb = {}
   local count = 0
   local gloveLines = 0
   for line in io.lines(path) do
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
      if (count == 1) then
         break
      end
   end

   io.write('done.','\n')
   return word_emb
end

function M.parseDataset(data_path,file_path)

   local index = {}
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
                     rev_index[vocab_size] = word
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
   return index, rev_index, vocab_size, len_max, total_lines

end

function M.load(d)

   print("Parsing and Writing Data")

   local f = io.open(d.saved_vocab_path,'r')
   local g = io.open(d.file_path,'r')
   if f == nil or g == nil then
      d.index, d.rev_index, d.vocab_size, d.len_max, d.total_lines = M.parseDataset(d.data_path,d.file_path)
      print("Saving Vocab")
      torch.save(d.saved_vocab_path,{d.index,d.rev_index,d.vocab_size,d.len_max,d.total_lines})
   else
      io.close(f)
      d.index, d.rev_index, d.vocab_size, d.len_max, d.total_lines = unpack(torch.load(d.saved_vocab_path))
   end

   print("Vocab Size " .. d.vocab_size)
   print("Len Max " .. d.len_max)
   print("Total Lines " .. d.total_lines)

   print("Retrieve Word Embeddings")
   -- Subtracting 1 from vocab_size because it includes <EOM>

   local f = io.open(d.saved_word_path,'r')
   if f == nil then
      d.word_emb = M.getWordEmbeddings(d.raw_word_path,d.index,d.vocab_size-1)
      print("Saving Embeddings")
      torch.save(d.saved_word_path,d.word_emb)
   else
      f:close()
      d.word_emb = torch.load(d.saved_word_path)
   end

   print("Generate Lookup Table")

   d.default_index = d.vocab_size + 1
   d.lookup_size = d.default_index
   d.lookup = torch.Tensor(d.lookup_size,d.dim)
   d.lookup[d.default_index] = torch.zeros(d.dim)

   for word,num in pairs(d.index) do
      if (d.word_emb[word] ~= nil) then
         d.lookup[num] = d.word_emb[word]
      else
         d.lookup[num] = torch.randn(d.dim)
      end
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
   enc_d.dim = 300
   enc_d.index = {}
   enc_d.rev_index = {}
   enc_d.word_emb = {}
   enc_d.lookup = {}
   enc_d.vocab_size = 0
   enc_d.len_max = 0
   enc_d.total_lines = 0
   if paths.filep(enc_d.file_path .. '.shuf') then 
      os.execute("rm " .. enc_d.file_path .. '.shuf')
   end
   if paths.filep(enc_d.file_path) then
      os.execute("rm " .. enc_d.file_path)
   end
   M.load(enc_d)

   print("\27[31mDecoder Data\n-------------")

   local dec_d = {}
   dec_d.base_path = opts.base_path
   dec_d.data_path =  opts.data_dir_from .. '/dec/'
   dec_d.file_path = opts.data_dir_to .. 'dec.txt'
   dec_d.raw_word_path = opts.glove_path
   dec_d.saved_word_path = opts.data_dir_to .. 'wordDec.th7'
   dec_d.saved_vocab_path = opts.data_dir_to .. 'vocabDec.th7'
   dec_d.dim = 300
   dec_d.index = {}
   dec_d.rev_index = {}
   dec_d.word_emb = {}
   dec_d.lookup = {}
   dec_d.vocab_size = 0
   dec_d.len_max = 0
   dec_d.total_lines = 0
   if paths.filep(dec_d.file_path .. '.shuf') then 
      os.execute("rm " .. dec_d.file_path .. '.shuf')
   end
   if paths.filep(dec_d.file_path .. '.shuf') then 
      os.execute("rm " .. dec_d.file_path)
   end
   M.load(dec_d)

   print("\27[31mPost Processing\n---------------")
   
   print("Shuffle Data")

   --os.execute("paste -d \':\' " .. enc_d.file_path .. ' ' .. dec_d.file_path .. " | shuf | awk -v FS=\":\" \'{ print $1 > \"" .. enc_d.file_path .. '.shuf' .. "\" ; print $2 > \"" .. dec_d.file_path .. '.shuf' .. "\" }\'")

   --enc_d.file_path = enc_d.file_path .. '.shuf'
   --dec_d.file_path = dec_d.file_path .. '.shuf'

   --cite author

   return enc_d, dec_d
end


