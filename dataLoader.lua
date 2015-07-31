require 'io'
require 'torch'
require 'reload'

local modname = ...
local M = {}
_G[modname] = M
package.loaded[modname] = M

function M.getWordEmbeddings(path,index)

   local word_emb = {}
   local count = 0
   
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
         print(count)
      end
   end
   return word_emb
end

function M.parseDataset(data_path,file_path)

   local index = {}
   local vocab_size = 0
   local len_max = 0
   local f = io.open(file_path,'w')
   for filename in io.popen('ls -a ' .. data_path):lines() do
      if filename ~= '.' and filename ~= '..' then
         print("Loading " .. filename)
         local data = io.open(data_path .. filename, 'r')
         for line in data:read('*a'):gmatch('.-[%.%?!]') do
            print(line)
            local len = 0
            for word in line:lower():gmatch("%a+") do
               f:write(word .. ' ')
               if (index[word] == nil) then
                  vocab_size = vocab_size + 1
                  index[word] = vocab_size
               end
               len = len + 1
            end
            if len > 0 then
               f:write('<EOS> \n')
            end
            if len > len_max then
               len_max = len
            end
         end
         data:close()
      end
   end
   if index['<EOS>'] == nil then
      vocab_size = vocab_size + 1
      index['<EOS>'] = vocab_size
   end
   f:close()
   return index, vocab_size, len_max

end

function M.load(d)

   print("Note: remove prev files if settings have changed")
   print("Note: current parser meant for prose")
   print("Parsing and Writing Data")

   local f = io.open(d.saved_vocab_path,'r')
   if f == nil then
      d.index, d.vocab_size, d.len_max = M.parseDataset(d.data_path,d.file_path)
      torch.save(d.saved_vocab_path,{d.index,d.vocab_size,d.len_max})
   else
      io.close(f)
      d.index, d.vocab_size, d.len_max = unpack(torch.load(d.saved_vocab_path))
   end

   print(d.index)
   print("Vocab Size " .. d.vocab_size)
   print("Len Max " .. d.len_max)

   print("Retrieve Word Embeddings")

   local f = io.open(d.saved_word_path,'r')
   if f == nil then
      d.word_emb = M.getWordEmbeddings(d.raw_word_path,d.index)
      torch.save(d.saved_word_path,word_emb)
   else
      io.close(f)
      d.word_emb = torch.load(d.saved_word_path)
   end

   print("Generate Lookup Table")

   d.lookup = torch.Tensor(count,dim)

   for word,num in d.index do
      if (d.word_emb[word] ~= nil) then
         d.lookup[num] = d.word_emb[word]
      else
         d.lookup[num] = torch.randn(d.dim)
      end
   end
end

function M.get()

   local base_path = '/deep/group/speech/asamar/nlp/seq/'
   local glove_path = '/deep/group/speech/asamar/nlp/glove/pretrained/glove.840B.300d.txt'
   local enc_d = {}
   enc_d.base_path = base_path
   enc_d.data_path =  '/deep/group/speech/asamar/nlp/data/gutenberg/txt/'
   enc_d.file_path = enc_d.base_path .. '/exp/data/enc.txt'
   enc_d.raw_word_path = glove_path
   enc_d.saved_word_path = enc_d.base_path .. '/exp/data/gloveEnc.th7'
   enc_d.saved_vocab_path = enc_d.base_path .. '/exp/data/vocabEnc.th7'
   enc_d.dim = 300
   enc_d.index = {}
   enc_d.word_emb = {}
   enc_d.lookup = {}
   enc_d.vocab_size = 0
   enc_d.len_max = 0
   os.execute("rm " .. enc_d.file_path .. '.shuf')
   os.execute("rm " .. enc_d.file_path)
   M.load(enc_d)

   local dec_d = {}
   dec_d.base_path = base_path
   dec_d.data_path =  '/deep/group/speech/asamar/nlp/data/gutenberg/txt/'
   dec_d.file_path = dec_d.base_path .. '/exp/data/dec.txt'
   dec_d.raw_word_path = glove_path
   dec_d.saved_word_path = dec_d.base_path .. '/exp/data/gloveDec.th7'
   dec_d.saved_vocab_path = dec_d.base_path .. '/exp/data/gutenberg/vocabDec.th7'
   dec_d.dim = 300
   dec_d.index = {}
   dec_d.word_emb = {}
   dec_d.lookup = {}
   dec_d.vocab_size = 0
   dec_d.len_max = 0
   os.execute("rm " .. dec_d.file_path .. '.shuf')
   os.execute("rm " .. dec_d.file_path)
   M.load(dec_d)

   print("Shuffle Data")

   os.execute("paste -d \':\' " .. enc_d.file_path .. ' ' .. dec_d.file_path .. " | shuf | awk -v FS=\":\" \'{ print $1 > \"" .. enc_d.file_path .. '.shuf' .. "\" ; print $2 > \"" .. dec_d.file_path .. '.shuf' .. "\" }\'")

   enc_d.file_path = enc_d.file_path .. '.shuf'
   dec_d.file_path = dec_d.file_path .. '.shuf'

   return enc_d, dec_d
end


