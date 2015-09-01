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

   return glove_vecs
end

function M.parse(data_file,opts)

   local index = {}
   local word_count = {}
   local rev_index = {}
   local vocab_size = 0
   local len_max = 0
   local total_lines = 0
   print("Loading  " .. data_file)
   local data = io.open(data_file, 'r')
   while true do
      local line = data:read()
      if line == nil then break end
      local words = stringx.split(line:lower())
      local len = words:len()
      if (len > 0) then 
         total_lines = total_lines + 1
         for _,word in ipairs(words) do
            if (index[word] == nil) then
               vocab_size = vocab_size + 1
               word_count[word] = 1
               table.insert(rev_index, word)
               index[word] = vocab_size
            else
               word_count[word] = word_count[word] + 1
            end
         end
         if len > len_max then
            len_max = len
         end
         old_line = line
      end
   end
   data:close()

   if index['<unk>'] == nil then
      vocab_size = vocab_size + 1
      table.insert(rev_index, '<unk>')
      index['<unk>'] = vocab_size
   end

   print("Total Lines " .. total_lines)
   print("Vocab Size " .. vocab_size)
   print("Max Sequence Length " .. len_max)

   for key,val in ipairs(rev_index) do
      index[val] = key
   end

   return index, rev_index, word_count, vocab_size, len_max, total_lines
end

function M.load(d,opts)

   if paths.filep(d.saved_vocab_file) == false then
      d.index, d.rev_index, d.word_count, 
      d.vocab_size, d.len_max, d.total_lines = 
         M.parse(d.train_file,opts)       
      print("Saving")
      torch.save(d.saved_vocab_file,
                 {d.index,d.rev_index,d.word_count, d.vocab_size,
                  d.len_max,d.total_lines})
   else
      print("Loading Saved Data")
      d.index, d.rev_index, d.word_count, 
      d.vocab_size, d.len_max, d.total_lines = 
         unpack(torch.load(d.saved_vocab_file))
   end

   if opts.glove then
      if paths.filep(d.saved_glove_file) == false then
         d.glove_vecs = {}
         if opts.glove then 
            print("Getting Glove Vecs")
         -- Subtracting 2 from vocabsize due to <EOS> and <unk>
            d.glove_vecs = M.getGlove(d.glove_file,d.index,d.vocab_size-2)
         end
         print("Saving Glove Vecs")
         torch.save(d.saved_glove_file,d.glove_vecs)
      else
         print("Loading Glove Vecs")
         d.glove_vecs = torch.load(d.saved_glove_file)
      end
   end

   print("Generating Lookup")
   
   d.default_index = d.vocab_size + 1 -- for the 'zero' lookup 
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

   -- Not saving this to keep debugging easy (randomness)

end

function M.get(opts)

   print("\27[31mEncoder Data\n-------------")

   local enc_d = {}
   enc_d.train_file =  opts.enc_train_file
   enc_d.test_file = opts.enc_test_file
   enc_d.glove_file = opts.glove_file
   enc_d.saved_glove_file = opts.run_dir .. '/gloveEnc.th7'
   enc_d.saved_vocab_file = opts.run_dir .. '/vocabEnc.th7'
   enc_d.saved_lookup_file = opts.run_dir .. '/lookupEnc.th7'
   enc_d.dim = opts.in_size
   enc_d.index = {}
   enc_d.rev_index = {}
   enc_d.word_emb = {}
   enc_d.lookup = {}
   enc_d.vocab_size = 0
   enc_d.len_max = 0
   enc_d.total_lines = 0
   enc_d.high_freq = 0
   if paths.filep(enc_d.train_file .. '.shuf') then 
      os.execute("rm " .. enc_d.train_file .. '.shuf')
   end
   M.load(enc_d,opts)

   print("\27[31mDecoder Data\n-------------")

   local dec_d = {}
   dec_d.train_file =  opts.dec_train_file
   dec_d.test_file = opts.dec_test_file
   dec_d.glove_file = opts.glove_file
   dec_d.saved_glove_file = opts.run_dir .. '/gloveDec.th7'
   dec_d.saved_vocab_file = opts.run_dir .. '/vocabDec.th7'
   dec_d.saved_lookup_file = opts.run_dir .. '/lookupDec.th7'
   dec_d.dim = opts.in_size
   dec_d.index = {}
   dec_d.rev_index = {}
   dec_d.word_emb = {}
   dec_d.lookup = {}
   dec_d.vocab_size = 0
   dec_d.len_max = 0
   dec_d.total_lines = 0
   dec_d.high_freq = 0
   if paths.filep(dec_d.train_file .. '.shuf') then 
      os.execute("rm " .. dec_d.train_file .. '.shuf')
   end
   M.load(dec_d,opts)

   -- print("\27[31mPost Processing\n---------------")  
   -- print("Shuffle Data")

   --os.execute("paste -d \':\' " .. enc_d.file_path .. ' ' 
   -- .. dec_d.file_path .. " | shuf | awk -v FS=\":\" \'{ print $1 > \"" .. 
   -- enc_d.file_path .. '.shuf' .. "\" ; print $2 > \"" .. 
   -- dec_d.file_path .. '.shuf' .. "\" }\'")

   --enc_d.file_path = enc_d.file_path .. '.shuf'
   --dec_d.file_path = dec_d.file_path .. '.shuf'

   --cite author

   return enc_d, dec_d
end


