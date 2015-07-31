require 'nn'

local DecCriterion, parent = torch.class('DecCriterion', 'nn.ClassNLLCriterion')

function DecCriterion:__init(weights)
   parent.__init(self,weights)
end


function DecCriterion:__len()
   return parent.__len()
end


function DecCriterion:updateOutput(input, target)
   if target == 0 then
      self.output:zero()
      return self.output
   else
      return parent.updateOutput(input,target)
   end
end

function DecCriterion:updateGradInput(input, target)
   if target == 0 then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      return self.gradInput
   else
      return parent.updateGradInput(input, target)
   end
end
