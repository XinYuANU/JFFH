------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cudnn'
require 'PerceptionLoss'
require 'preprocess'
require 'TripletLoss_newest'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
URnet = require 'adverserial_xin_v1_D_revise_perception_triplet_alternative'
--URnet = require 'UR_rgb'
require 'stn_L1'
require 'stn_L2'
require 'stn_L3'
require 'stn_L4'
require 'stn_L5'
dl = require 'dataload'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs_Triplet_noskip_newest_A")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 30)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  --scale            (default 128)          scale of images to train on
  --lambda           (default 0.01)       trade off D and Euclidean distance 
  --eta   	         (default 0.01)       trade off G and perception loss  
  --margin           (default 0.3)        trade off D and G 
  --triloss          (default 0.0001)		  the loss of embedding 
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end
opt.save = opt.save .. string.format('_TriLoss_%.4f', opt.triloss)

print(opt)

ntrain = 41700
ntrain_id = 8340
nval   = 1024

--local highHd5 = hdf5.open('datasets/YTC_HR_Front_NF.hdf5', 'r')
--local data_HR = highHd5:read('YTC'):all()
--data_HR:mul(2):add(-1)
--highHd5:close()
--trainData_HR = data_HR[{{1, ntrain}}]
--valData_HR = data_HR[{{ntrain+1, nval+ntrain}}]

datapath = 'datasets'
loadsize = {3, opt.scale, opt.scale}
nthread  = opt.threads
-- train_dataset
train_filename_HR = 'File_order.txt'
train_imagefolder_HR = 'HR_NF'
print('loading training HR')
train_HR = dl.ImageClassPairs(datapath, train_filename_HR, train_imagefolder_HR, loadsize)


local lowHd5 = hdf5.open('datasets/YTC_LR_Side.hdf5', 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR[{{1, ntrain}}]
valData_LR = data_LR[{{ntrain+nval+2, 2*nval+ntrain+1}}]

-- here is the differences between sampling from images and hdf5
--tmp = train_HR:index(torch.LongTensor{1})
--tmp_LR = data_LR[1]--:index(1,torch.LongTensor{3,2,1})
--tmp_LR:add(1):mul(0.5)
--tmp = tmp:index(2, torch.LongTensor{3,2,1})
--print(tmp:size(), tmp_LR:size(),tmp[1][1][1][1], tmp_LR[1][1][1])
--image.save('hr.png',tmp[1])
--image.save('lr.png',tmp_LR)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
	cutorch.setDevice(opt.gpu + 1)
	print('<gpu> using device ' .. opt.gpu)
	torch.setdefaulttensortype('torch.CudaTensor')
else
	torch.setdefaulttensortype('torch.FloatTensor')
end

input_scale  = 16
print(opt.scale)
opt.geometry = {3, opt.scale, opt.scale}

if opt.network == '' then
  
	model_D = nn.Sequential()
	model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.SpatialMaxPooling(2,2))
	model_D:add(cudnn.ReLU(true))
	model_D:add(nn.SpatialDropout(0.2))  
	model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.SpatialMaxPooling(2,2))
	model_D:add(cudnn.ReLU(true))
	model_D:add(nn.SpatialDropout(0.2))
	model_D:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.SpatialMaxPooling(2,2))
	model_D:add(cudnn.ReLU(true))
	model_D:add(nn.SpatialDropout(0.2))
	model_D:add(cudnn.SpatialConvolution(128, 96, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.ReLU(true))
	model_D:add(cudnn.SpatialMaxPooling(2,2))
	model_D:add(nn.SpatialDropout(0.2))
	model_D:add(nn.Reshape(8*8*96))
	model_D:add(nn.Linear(8*8*96, 1024))
	model_D:add(cudnn.ReLU(true))
	model_D:add(nn.Dropout())
	model_D:add(nn.Linear(1024,1))
	model_D:add(nn.Sigmoid()) 
	
	----------------------------------------------------------------------
	local input = nn.Identity()()
	local e1 = input - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true) - stn_L1(32)  --16  
	local e2 = e1 - cudnn.SpatialMaxPooling(2,2) - cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - stn_L2(128)--8
	local e3 = e2 - cudnn.SpatialMaxPooling(2,2) - cudnn.SpatialConvolution(128, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true)-- 4
	local e4 = e3 - cudnn.SpatialMaxPooling(2,2) --2
	
	local e5 = e4 - nn.Reshape(2*2*512) - nn.Linear(2*2*512,2*2*512) 
	
	local d1 = e5 - nn.View(512,2,2) - cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) --2
	local d2 = d1 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true)-- 4*4
	local d3 = d2 - nn.SpatialUpSamplingNearest(2) - stn_L3(128) - cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true)-- 8
	local d4 = d3 - nn.SpatialUpSamplingNearest(2) - stn_L4(256) - cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) -- 16
	local d5 = d4 - nn.SpatialUpSamplingNearest(2) - stn_L5(512) - cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true)  --32
	local d6 = d5 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) -- 64
	local d7 = d6 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) -- 128
	local d8 = d7 - cudnn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2)

	model_G = nn.gModule({input},{e5, d8})

else
	print('<trainer> reloading previously trained network: ' .. opt.network)
	tmp = torch.load(opt.network)
	model_D = tmp.D  
	model_G = tmp.G
	epoch = 110  
end

print('Copy model to gpu')
model_D:cuda()
model_G:cuda()  -- convert model to CUDA

-- loss function: negative log-likelihood
criterion_D = nn.BCECriterion()
criterion_G = nn.MSECriterion()
criterion_T = nn.TripletLoss()

vgg_model = createVggmodel()
PerceptionLoss = nn.PerceptionLoss(vgg_model, 1):cuda()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Training parameters
sgdState_D = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	optimize=true,
	numUpdates = 0
}
sgdState_G = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	optimize=true,
	numUpdates=0
}

-- Get examples to plot
function getSamples(dataset_LR, N)
	local N = N or 10
	local inputs   = torch.Tensor(N,3,16,16)
	for i = 1,N do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_HR[i]),16,16)
		inputs[i] = dataset_LR[i]
	end
	
	local samples = model_G:forward(inputs)
	samples = nn.HardTanh():forward(samples[2])
	torch.setdefaulttensortype('torch.FloatTensor')
	inputs_HR = train_HR:index(torch.range(ntrain+nval+2, ntrain+nval+1+N))
	inputs_HR:mul(2):add(-1)
	inputs_HR = inputs_HR:index(2, torch.LongTensor{3,2,1})
	local to_plot = {}
	for i = 1,N do 
		to_plot[#to_plot+1] = samples[i]:float()
		if i % 5 == 0 then
			to_plot[#to_plot+1] = inputs_HR[i]:float()
		end
		
	end
	torch.setdefaulttensortype('torch.CudaTensor')
	return to_plot
end


function generateRandomSamples( num_id )
	-- generate all the triplet
	local idx = torch.randperm(num_id)
	local all_triplets = torch.Tensor(3*num_id*4):zero()

	for i = 1, num_id do
		local anchor = 0
		local neg = 0

		if i == num_id then 
			anchor  = (idx[i]-1)*5+3
			neg = (idx[1]-1)*5+3
		else
			anchor = (idx[i]-1)*5+3
			neg = (idx[i+1]-1)*5+3
		end

		local t = {-2, -1, 1, 2}
		for j = 1, 4 do
			all_triplets[(i-1)*12 + 3*(j-1) +1 ] = anchor + t[j]
			all_triplets[(i-1)*12 + 3*(j-1) +2 ] = anchor 
			all_triplets[(i-1)*12 + 3*(j-1) +3 ] = neg	
		end
	end

	local idx_triplets = torch.randperm(num_id*4) 
	local all_triplets_perm = torch.Tensor(3*num_id*4):zero()
	for i = 1, num_id*4 do
		all_triplets_perm[{{ (i-1)*3+1, i*3 }}] = all_triplets[{{ (idx_triplets[i]-1)*3+1,  idx_triplets[i]*3 }}]:clone()
	end

	return all_triplets_perm
end


while true do 
	local to_plot = getSamples(valData_LR, 100)
	torch.setdefaulttensortype('torch.FloatTensor')
	
	
	local formatted = image.toDisplayTensor({input = to_plot, nrow = 12})
	formatted:float()
	formatted = formatted:index(1,torch.LongTensor{3,2,1})
	
	image.save(opt.save .. '/UR_example_' .. (epoch or 0) .. '.png', formatted)
	
	IDX = generateRandomSamples(ntrain_id)

	if opt.gpu then 
		torch.setdefaulttensortype('torch.CudaTensor')
	else
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	
	URnet.train(trainData_LR,train_HR,ntrain_id*12)
	
	sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.00001)
	
	sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
	sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.00001)
	
	opt.lambda = math.max(opt.lambda*0.995, 0.005)   -- or 0.995
--	opt.eta    = math.max(opt.eta*1.05, 0.1)   -- or 0.995
	
	--opt.triloss = math.min(opt.triloss*1.05, 0.0001)
end
