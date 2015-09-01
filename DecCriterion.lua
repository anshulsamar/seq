require 'nn'

local DecCriterion, parent = torch.class('DecCriterion', 'nn.ClassNLLCriterion')

function DecCriterion:__init(weights)
   parent.__init(self,weights)
end


function DecCriterion:__len()
   return parent.__len(self)
end


function DecCriterion:updateOutput(input, target)
   table.insert(dec_output,torch.exp(input))
   --print(target)
   --print(input)
   --print(torch.exp(input))
   if target == 0 then
      self.output:zero()
      return self.output
   else
      local ret = parent.updateOutput(self,input,target)
      --print('Error ' .. ret)
      return ret
   end
end

function DecCriterion:updateGradInput(input, target)
   if target == 0 then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      return self.gradInput
   else
      return parent.updateGradInput(self,input, target)
   end
end
