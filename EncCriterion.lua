local EncoderCriterion, parent = torch.class('EncoderCriterion', 'nn.ClassNLLCriterion')

function EncoderCriterion:__init(weights)
   parent.__init(weights)
end

function EncoderCriterion:updateOutput(input, target)
   self.output:zero()
   return self.output
end

function EncoderCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   return self.gradInput
end
