require 'nn'

local EncCriterion, parent = torch.class('EncCriterion', 'nn.ClassNLLCriterion')

function EncCriterion:__init(weights)
   parent.__init(self,weights)
end

function EncCriterion:updateOutput(input, target)
   self.output:zero()
   return self.output
end

function EncCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   return self.gradInput
end
