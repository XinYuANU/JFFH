require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn_L1'
require 'stn_L2'
require 'stn_L3'
require 'stn_L4'
require 'stn_L5'
require 'stn_L6'


local Model = {}

local conv = cudnn.SpatialConvolution
local batchnorm = cudnn.SpatialBatchNormalization
local relu = cudnn.ReLU
local upsample = nn.SpatialUpSamplingNearest


local function convBlock(inChannels, outChannels)
    local convnet = nn.Sequential()
    convnet:add(cudnn.SpatialBatchNormalization(inChannels))
    convnet:add(cudnn.ReLU(true))
    convnet:add(cudnn.SpatialConvolution(inChannels, outChannels, 3, 3, 1, 1, 1, 1))
    convnet:add(cudnn.SpatialBatchNormalization(outChannels))
    convnet:add(cudnn.ReLU(true))
    convnet:add(cudnn.SpatialConvolution(outChannels, outChannels, 1, 1))
    return convnet
end


-- Skip layer
local function skipLayer(numIn, numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end


-- Residual block 
function Residual(numIn, numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end


local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return cudnn.ReLU(true)(cudnn.SpatialBatchNormalization(numOut)(l))
end



function Model.createNetD()
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

    return model_D
end

----[[
function Model.createNetG()
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true) - stn_L1(32) - cudnn.SpatialMaxPooling(2,2) --8*8 

    local e2 = e1 - cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - stn_L2(128) - cudnn.SpatialMaxPooling(2,2)-- 4*4

    local e3 = e2 - cudnn.SpatialConvolution(128, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - cudnn.SpatialMaxPooling(2,2) --2*2

    local e4 = e3 - nn.Reshape(2*2*512) - nn.Linear(2*2*512, 2*2*512)

    local e5 = e4 - cudnn.ReLU(true) - nn.View(512, 2, 2)

    local e6 = e5 - cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 4*4

    local e7 = e6 - cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L3(128) -- 8*8
    
    local e8 = e7 - cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L4(64) -- 16*16

    local e9 = e8 - cudnn.SpatialConvolution(64, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L5(512) -- 32*32

    local e10 = e9 - cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) --64*64

    local e11 = e10 - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 128*128

    local e12 = e11 - cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - cudnn.SpatialConvolution(64, 16, 3, 3, 1, 1, 1, 1) - cudnn.ReLU(true) - cudnn.SpatialConvolution(16, 3, 5, 5, 1, 1, 2, 2)

    model_G = nn.gModule({input}, {e4, e12})
    return model_G
end

function Model.createNetG_wotri()
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true) - stn_L1(32) - cudnn.SpatialMaxPooling(2,2) --8*8 

    local e2 = e1 - cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - stn_L2(128) - cudnn.SpatialMaxPooling(2,2)-- 4*4

    local e3 = e2 - cudnn.SpatialConvolution(128, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - cudnn.SpatialMaxPooling(2,2) --2*2

    local e4 = e3 - nn.Reshape(2*2*512) - nn.Linear(2*2*512, 2*2*512)

    local e5 = e4 - cudnn.ReLU(true) - nn.View(512, 2, 2)

    local e6 = e5 - cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 4*4

    local e7 = e6 - cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L3(128) -- 8*8
    
    local e8 = e7 - cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L4(64) -- 16*16

    local e9 = e8 - cudnn.SpatialConvolution(64, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L5(512) -- 32*32

    local e10 = e9 - cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) --64*64

    local e11 = e10 - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 128*128

    local e12 = e11 - cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - cudnn.SpatialConvolution(64, 16, 3, 3, 1, 1, 1, 1) - cudnn.ReLU(true) - cudnn.SpatialConvolution(16, 3, 5, 5, 1, 1, 2, 2)

    model_G = nn.gModule({input}, {e12})
    return model_G
end


function Model.createNetG_residual()
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true) - stn_L1(32) - cudnn.SpatialMaxPooling(2,2) --8*8 

    local e2 = e1 - cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - stn_L2(128) - cudnn.SpatialMaxPooling(2,2)-- 4*4

    local e3 = e2 - cudnn.SpatialConvolution(128, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - cudnn.SpatialMaxPooling(2,2) --2*2

    local e4 = e3 - nn.Reshape(2*2*512) - nn.Linear(2*2*512, 2*2*512)

    local e5 = e4 - cudnn.ReLU(true) - nn.View(512, 2, 2)

    local e6 = e5 - cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 4*4

    local e7 = e6 - cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L3(128) -- 8*8
    
    local e8 = e7 - cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L4(64) -- 16*16

    local e9 = e8 - cudnn.SpatialConvolution(64, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L5(512) -- 32*32

    local e10 = e9 - cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) --64*64

    local e11 = e10 - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - Residual(128, 128) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 128*128

    local e12 = e11 - cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2) - Residual(64, 64) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - cudnn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1) 
    local e13 = e12 - Residual(32, 16) - cudnn.ReLU(true) - cudnn.SpatialConvolution(16, 3, 5, 5, 1, 1, 2, 2)

    model_G = nn.gModule({input}, {e4, e13})
    return model_G
end


function Model.createNetG_relu()
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true) - stn_L1(32) - cudnn.SpatialMaxPooling(2,2) --8*8 

    local e2 = e1 - cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - stn_L2(128) - cudnn.SpatialMaxPooling(2,2)-- 4*4

    local e3 = e2 - cudnn.SpatialConvolution(128, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - cudnn.SpatialMaxPooling(2,2) --2*2

    local e4 = e3 - nn.Reshape(2*2*512) - nn.Linear(2*2*512, 2*2*512) - cudnn.ReLU(true)

    local e5 = e4 - nn.View(512, 2, 2)

    local e6 = e5 - cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 4*4

    local e7 = e6 - cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L3(128) -- 8*8
    
    local e8 = e7 - cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L4(64) -- 16*16

    local e9 = e8 - cudnn.SpatialConvolution(64, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L5(512) -- 32*32

    local e10 = e9 - cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) --64*64

    local e11 = e10 - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 128*128

    local e12 = e11 - cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - cudnn.SpatialConvolution(64, 16, 3, 3, 1, 1, 1, 1) - cudnn.ReLU(true) - cudnn.SpatialConvolution(16, 3, 5, 5, 1, 1, 2, 2)

    model_G = nn.gModule({input}, {e4, e12})
    return model_G
end


function Model.createNetG_relu_residual()
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true) - stn_L1(32) - cudnn.SpatialMaxPooling(2,2) --8*8 

    local e2 = e1 - cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - stn_L2(128) - cudnn.SpatialMaxPooling(2,2)-- 4*4

    local e3 = e2 - cudnn.SpatialConvolution(128, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - cudnn.SpatialMaxPooling(2,2) --2*2

    local e4 = e3 - nn.Reshape(2*2*512) - nn.Linear(2*2*512, 2*2*512) - cudnn.ReLU(true)

    local e5 = e4 - nn.View(512, 2, 2)

    local e6 = e5 - cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 4*4

    local e7 = e6 - cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L3(128) -- 8*8
    
    local e8 = e7 - cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L4(64) -- 16*16

    local e9 = e8 - cudnn.SpatialConvolution(64, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - stn_L5(512) -- 32*32

    local e10 = e9 - cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) --64*64

    local e11 = e10 - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - Residual(128, 128) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 128*128

    local e12 = e11 - cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2) - Residual(64, 64) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true) - cudnn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1) 
    local e13 = e12 - Residual(32, 16) - cudnn.ReLU(true) - cudnn.SpatialConvolution(16, 3, 5, 5, 1, 1, 2, 2)

    model_G = nn.gModule({input}, {e4, e13})
    return model_G
end
--]]


function Model.createNetG_residual_norm(ngf)
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) - stn_L1(ngf*2) -- 16*16  32

    local e2 = e1 - cudnn.SpatialConvolution(ngf*2, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true) -- 8*8  64

    local e3 = e2 - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 4*4
    
    local e4 = e3 - cudnn.SpatialConvolution(ngf*16, ngf*64, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - cudnn.ReLU(true) -- 2*2
    
    local e5 = e4 - cudnn.SpatialConvolution(ngf*64, ngf*128, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*128) - nn.Reshape(ngf*128) -- 1*1

    local e6 = e5 - nn.View(ngf*128, 1, 1) 
    
    local e7 = e6 - cudnn.SpatialFullConvolution(ngf*128, ngf*128, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*128) - cudnn.ReLU(true)-- 2*2
    
    local e8 = e7 - cudnn.SpatialFullConvolution(ngf*128, ngf*64, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - cudnn.ReLU(true) -- 4*4
  
    local e9 = e8 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true) -- 8*8

    local e10 = e9 - cudnn.SpatialFullConvolution(ngf*32, ngf*32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true) -- 16*16

    local e11 = e10 - cudnn.SpatialFullConvolution(ngf*32, ngf*32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true) -- - stn_L5(ngf*32) --32*32

    local e12 = e11 - cudnn.SpatialFullConvolution(ngf*32, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true) - stn_L6(ngf*16) --64*64
    
    local e13 = e12 - cudnn.SpatialFullConvolution(ngf*16, ngf*8, 4, 4, 2, 2, 1, 1)- cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true) --128*128

    local e14 = e13 - cudnn.SpatialConvolution(ngf*8, ngf*4, 7, 7, 1, 1, 3, 3) - Residual(ngf*4, ngf*2) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf*2, 3, 5, 5, 1, 1, 2, 2) 
    
    model_G = nn.gModule({input}, {e5, e14})
    return model_G
end


function Model.createNetG_residual_norm_org(ngf)
    input = nn.Identity()()
    local e1 = input - cudnn.SpatialConvolution(3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) - stn_L1(ngf*2) -- 16*16  32

    local e2 = e1 - cudnn.SpatialConvolution(ngf*2, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true) -- 8*8  64

    local e3 = e2 - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 4*4
    
    local e4 = e3 - cudnn.SpatialConvolution(ngf*16, ngf*64, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - cudnn.ReLU(true) -- 2*2
    
    local e5 = e4 - cudnn.SpatialConvolution(ngf*64, ngf*128, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*128) - nn.Reshape(ngf*128) -- 1*1

    local e6 = e5 - nn.View(ngf*128, 1, 1) 
    
    local e7 = e6 - cudnn.SpatialFullConvolution(ngf*128, ngf*128, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*128) - cudnn.ReLU(true)-- 2*2
    
    local e8 = e7 - cudnn.SpatialFullConvolution(ngf*128, ngf*64, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - cudnn.ReLU(true) -- 4*4
  
    local e9 = e8 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true) -- 8*8

    local e10 = e9 - cudnn.SpatialFullConvolution(ngf*32, ngf*32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true) -- 16*16

    local e11 = e10 - cudnn.SpatialFullConvolution(ngf*32, ngf*32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true) -- - stn_L5(ngf*32) --32*32

    local e12 = e11 - cudnn.SpatialFullConvolution(ngf*32, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true) - stn_L6(ngf*16) --64*64
    
    local e13 = e12 - cudnn.SpatialFullConvolution(ngf*16, ngf*8, 4, 4, 2, 2, 1, 1)- cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true) --128*128

    local e14 = e13 - cudnn.SpatialConvolution(ngf*8, ngf*4, 7, 7, 1, 1, 3, 3) - Residual(ngf*4, ngf*2) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf*2, 3, 5, 5, 1, 1, 2, 2) 
    
    model_G = nn.gModule({input}, {e14})
    return model_G
end

return Model