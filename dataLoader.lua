require 'io'
require 'torch'

local modname = ...
local M = {}
_G[modname] = M
package.loaded[modname] = M

function M.getWordEmbeddings(path)

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
      end
   end
   return word_emb
end

function M.parseDataset(data_path,data_file)

   local index = {}
   local vocab_size = 0
   local len_max = 0
   local f = io.open(data_file,'w')
   for filename in io.popen('ls -a ' .. data_path):lines() do
      if filename ~= '.' and filename ~= '..' then
         print("Loading " .. filename)
         for line in io.lines(data_path .. filename) do
            local len = 0
            for word in line:lower():gmatch("%a+") do
               f:write(word .. ' ')
               if (index[word] == nil) then
                  vocab_size += 1
                  index[word] = count
               end
               len = len + 1
            end
            f:write('<EOS> \n')
            if len > len_max then
               len_max = len
            end
         end
      end
   end
   if index['<EOS>'] == nil then
      vocab_size += 1
      index['<EOS>'] = vocab_size
   end
   f:close()
   return index, vocab_size, len_max

end

function M.load(d)

   print("Parsing and Writing Data")

   local f = io.open(enc.saved_vocab_path,'r')
   if f == nil then
      d.index, d.vocab_size, d.len_max = parseDataset(d.data_path,d.file_path)
      torch.save(d.saved_vocab_path,{d.index,d.vocab_size,d.len_max})
   else
      io.close(f)
      d.index, d.vocab_size, d.len_max = unpack(torch.load(d.saved_vocab_path))
   end

   print("Shuffle Data")

   os.execute("shuf " .. d.file_path .. " -o " .. d.file_path .. "tmp.txt")
   os.execute("rm " .. d.file_path)
   os.execute("mv " .. d.file_path .. "tmp.txt " .. d.file_path)

   print("Retrieve Word Embeddings")

   local f = io.open(d.saved_word_path,'r')
   if f == nil then
      d.word_emb = M.getWordEmbeddings(d.raw_word_path)
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

   local glove_path = '/glove/pretrained/glove.840B.300d.txt'
   local base_path = '/deep/group/speech/asamar/nlp/'
   local enc_d = {}
   enc_d.base_path = base_path
   enc_d.data_path =  enc_d.base_path .. '/data/gutenberg/txt'
   enc_d.file_path = enc_d.base_path .. '/exp/data/enc.txt'
   enc_d.raw_word_path = enc_d.base_path .. glove_path
   enc_d.saved_word_path = enc_d.base_path .. '/exp/data/gloveEnc.th7'
   enc_d.saved_vocab_path = enc_d.base_path .. '/data/gutenberg/vocabEnc.th7'
   enc_d.dim = 300
   enc_d.index = {}
   enc_d.word_emb = {}
   enc_d.lookup = {}
   enc_d.vocab_size = 0
   enc_d.len_max = 0
   M.load(enc_d)

   local dec_d = {}
   dec_d.base_path = base_path
   dec_d.data_path =  dec_d.base_path .. '/data/gutenberg/txt'
   dec_d.file_path = dec_d.base_path .. '/exp/data/dec.txt'
   dec_d.raw_word_path = dec_d.base_path .. glove_path
   dec_d.saved_word_path = dec_d.base_path .. '/exp/data/gloveDec.th7'
   dec_d.saved_vocab_path = dec_d.base_path .. '/data/gutenberg/vocabDec.th7'
   dec_d.dim = 300
   dec_d.index = {}
   dec_d.word_emb = {}
   dec_d.lookup = {}
   dec_d.vocab_size = 0
   dec_d.len_max = 0
   M.load(dec_d)

   return enc_d, dec_d
end


