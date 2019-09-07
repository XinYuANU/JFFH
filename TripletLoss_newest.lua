--------------------------------------------------------------------------------
-- TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------
require 'nn'
require 'torch'

local TripletLoss, parent = torch.class('nn.TripletLoss', 'nn.Criterion')

function TripletLoss:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 0
   self.Li = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function TripletLoss:updateOutput(input)
   local len = input:size(1)
   local len_part = len/3 -- here we need to make sure the minibatch is able to be divided by three
   local a = input[{{1, len_part}}] -- anchor
   local p = input[{{len_part+1, len_part*2}}] -- positive
   local n = input[{{len_part*2+1, len_part*3}}] -- negative

--   local a_norm = torch.norm(a, 2, 2)
--   local p_norm = torch.norm(p, 2, 2)
--   local n_norm = torch.norm(n, 2, 2)

--   a:cdiv(a_norm:repeatTensor(1, a:size(2)) ):type(a:type())
--   p:cdiv(p_norm:repeatTensor(1, p:size(2)) ):type(p:type())
--   n:cdiv(n_norm:repeatTensor(1, n:size(2)) ):type(n:type())

   local N = a:size(1)
   self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)) , (a - p):norm(2,2):pow(2) -  (a - n):norm(2,2):pow(2) + self.alpha, 2), 2)
   self.output = self.Li:sum() / N
   return self.output
end

function TripletLoss:updateGradInput(input)
   local len = input:size(1)
   local len_part = len/3 -- here we need to make sure the minibatch is able to be divided by three
   local a = input[{{1, len_part}}] -- ancor
   local p = input[{{len_part+1, len_part*2}}] -- positive
   local n = input[{{len_part*2+1, len_part*3}}] -- negative

--   local a_norm = torch.norm(a, 2, 2)
--   local p_norm = torch.norm(p, 2, 2)
--   local n_norm = torch.norm(n, 2, 2)

--   a:cdiv(a_norm:repeatTensor(1, a:size(2)) ):type(a:type())
--   p:cdiv(p_norm:repeatTensor(1, p:size(2)) ):type(p:type())
--   n:cdiv(n_norm:repeatTensor(1, n:size(2)) ):type(n:type())

   local N = a:size(1)
   local dim = a:size(2)

   self.gradInput:resizeAs(input):zero()
   self.gradInput[{{1, len_part}}] = (a - p):cmul(self.Li:gt(0):repeatTensor(1,a:size(2)):type(a:type()) * 2/N/dim) --:cmul(a_norm:repeatTensor(1, a:size(2)) ):type(a:type())
   --self.gradInput[{{len_part+1, len_part*2}}] = (p - a):cmul(self.Li:gt(0):repeatTensor(1,a:size(2)):type(a:type()) * 2/N)  --:cmul(p_norm:repeatTensor(1, p:size(2)) ):type(p:type())
   --self.gradInput[{{len_part*2+1, len_part*3}}] = (a - n):cmul(self.Li:gt(0):repeatTensor(1,a:size(2)):type(a:type()) * 2/N)  --:cmul(n_norm:repeatTensor(1, n:size(2)) ):type(n:type())

   return self.gradInput
end