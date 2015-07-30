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

function M.parseDataset(data_path,encoder_data_file,decoder_data_file)

   local index = {}
   local vocab_size = 0
   local len_max = 0
   local enc_f = io.open(encoder_data_file,'w')
   local dec_f = io.open(decoder_data_file,'w')
   for filename in io.popen('ls -a ' .. rawDataPath):lines() do
      if filename ~= '.' and filename ~= '..' then
         print("Loading " .. filename)
         for line in io.lines(rawDataPath .. filename) do
            local len = 0
            for word in line:lower():gmatch("%a+") do
               enc_f:write(word .. ' ')
               dec_f:write(word .. ' ')
               if (index[word] == nil) then
                  vocab_size += 1
                  index[word] = count
               end
               len = len + 1
            end
            enc_f:write('<EOS> \n')
            dec_f:write('<EOS> \n')
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

function M.load()

   local encoder_data_file = './data/gutenberg/txt/'
   local word_file = './glove/pretrained/glove.840B.300d.txt'
   local saved_word_file = './glove/pretrained/glove.th7'
   local saved_vocab_file = './data/gutenberg/vocab.th7'
   local dim = 300
   local vocab_size = 0
   local index = {}
   local word_emb = {}
   local lookup = {}
   local len_max = 0

   local f = io.open(opts.saved_vocab_file,'r')
   if f == nil then
      index, vocab_size, len_max = parseDataset(data_path,encoder_data_file)
      torch.save(opts.saved_vocab_file,{vocab,count,maxLen})
   else
      io.close(f)
      index, vocab_size, len_max = unpack(torch.load(opts.saved_vocab_file))
   end

   local f = io.open(opts.saved_glove_file,'r')
   if f == nil then
      word_emb = M.getWordEmbeddings(word_file)
      torch.save(saved_word_file,word_emb)
   else
      io.close(f)
      word_emb = torch.load(saved_word_file)
   end

   lookup = torch.Tensor(count,dim)

   for word,num in index do
      if (word_emb[word] ~= nil) then
         lookup[num] = word_emb[word]
      else
         lookup[num] = torch.randn(dim)
      end
   end

   return {'enc_len_max'=len_max, 'enc_lookup'=lookup, 'enc_index'=index, 'enc_vocab_size'=vocab_size, 'dec_len_max'=len_max, 'dec_lookup'=lookup, 'dec_index'=index, 'dec_vocab_size'=vocab_size}

end



