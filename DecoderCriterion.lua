local DecoderCriterion, parent = torch.class('DecoderCriterion', 'nn.ClassNLLCriterion')

function DecoderCriterion:__init(weights)
   parent.__init(weights)
end


function DecoderCriterion:__len()
   return parent.__len()
end


function DecoderCriterion:updateOutput(input, target)
   if target == 0 then
      self.output:zero()
      return self.output
   else
      return parent.updateOutput(input,target)
   end
end

function DecoderCriterion:updateGradInput(input, target)
   if target == 0 then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      return self.gradInput
   else
      return parent.updateGradInput(input, target)
   end
end
