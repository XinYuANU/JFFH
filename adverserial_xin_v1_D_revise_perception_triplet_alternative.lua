require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'pl'
require 'dataload'

local adversarial = {}
local input_scale = 16
function rmsprop(opfunc, x, config, state)
	
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.9
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
        -- (2) initialize mean square values and square gradient storage
        if not state.m then
          state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
          state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end

        -- (3) calculate new (leaky) mean squared values
        state.m:mul(alpha)
        state.m:addcmul(1.0-alpha, dfdx, dfdx)

        -- (4) perform update
        state.tmp:sqrt(state.m):add(epsilon)
        -- only opdate when optimize is true
        
        
  	if config.numUpdates < 50 then
  	      --io.write(" ", lr/50.0, " ")
  	      x:addcdiv(-lr/50.0, dfdx, state.tmp)
  	elseif config.numUpdates < 100 then
  	    --io.write(" ", lr/5.0, " ")
  	    x:addcdiv(-lr /5.0, dfdx, state.tmp)
  	else 
  	  --io.write(" ", lr, " ")
  	  x:addcdiv(-lr, dfdx, state.tmp)
  	end
    end
    config.numUpdates = config.numUpdates +1
  

    -- return x*, f(x) before optimization
    return x, {fx}
end


function adam(opfunc, x, config, state)
    --print('ADAM')
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
	    -- Initialization
	    state.t = state.t or 0
	    -- Exponential moving average of gradient values
	    state.m = state.m or x.new(dfdx:size()):zero()
	    -- Exponential moving average of squared gradient values
	    state.v = state.v or x.new(dfdx:size()):zero()
	    -- A tmp tensor to hold the sqrt(v) + epsilon
	    state.denom = state.denom or x.new(dfdx:size()):zero()

	    state.t = state.t + 1
	    
	    -- Decay the first and second moment running average coefficient
	    state.m:mul(beta1):add(1-beta1, dfdx)
	    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

	    state.denom:copy(state.v):sqrt():add(epsilon)

	    local biasCorrection1 = 1 - beta1^state.t
	    local biasCorrection2 = 1 - beta2^state.t
	    
		local fac = 1
		if config.numUpdates < 10 then
		    fac = 50.0
		elseif config.numUpdates < 30 then
		    fac = 5.0
		else 
		    fac = 1.0
		end
		io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
	    -- (2) update x
	    x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end


function prepare_MiniBatch(indices)
    local len = indices:size(1)
    --print(indices:size())
    local num = len/3
    torch.setdefaulttensortype('torch.FloatTensor')

    local indices_new = torch.LongTensor(len):zero()
    for i = 1, num do
        indices_new[i] = indices[(i-1)*3+1] -- non-frontal pos	
        indices_new[i+num] = indices[(i-1)*3+2]  -- frontal pos	
        indices_new[i+2*num] = indices[(i-1)*3+3]  -- frontal neg
    end

    torch.setdefaulttensortype('torch.CudaTensor')
    return indices_new
end
-- training function

function adversarial.train(dataset_LR,dataset_HR, N)

  model_G:training()
  model_D:training()
  epoch = epoch or 0
  local N = N or ntrain
  N = math.floor(N/opt.batchSize)*opt.batchSize

  local dataBatchSize = opt.batchSize / 2
  local time = sys.clock()
  local err_gen = 0
  
  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local targets = torch.Tensor(opt.batchSize)

  local HR_inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local LR_inputs = torch.Tensor(opt.batchSize, opt.geometry[1], input_scale, input_scale)  
    
  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,N,opt.batchSize do --dataBatchSize do
    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
      local fevalD = function(x)
          collectgarbage()
          if x ~= parameters_D then -- get new parameters
            parameters_D:copy(x)
          end

          gradParameters_D:zero() -- reset gradients

          --  forward pass
          local outputs = model_D:forward(inputs)
          -- err_F = criterion_D:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
          -- err_R = criterion_D:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
          local err_R = criterion_D:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
          local err_F = criterion_D:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
        
          local margin = opt.margin -- org = 0.3
          sgdState_D.optimize = true
          sgdState_G.optimize = true      
          if err_F < margin or err_R < margin then
             sgdState_D.optimize = false
          end
          if err_F > (1.0-margin) or err_R > (1.0-margin) then
             sgdState_G.optimize = false
          end
          if sgdState_G.optimize == false and sgdState_D.optimize == false then
             sgdState_G.optimize = true 
             sgdState_D.optimize = true
          end
      
      
          --print(monA:size(), tarA:size())
          --io.write("v1_ytc| R:", err_R,"  F:", err_F, "  ")
          local f = criterion_D:forward(outputs, targets)

          -- backward pass 
          local df_do = criterion_D:backward(outputs, targets)
          model_D:backward(inputs, df_do)

          -- penalties (L1 and L2):
          if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
              local norm,sign= torch.norm,torch.sign
              -- Loss:
              f = f + opt.coefL1 * norm(parameters_D,1)
              f = f + opt.coefL2 * norm(parameters_D,2)^2/2
              -- Gradients:
              gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
          end

          --print('grad D', gradParameters_D:norm())
          return f,gradParameters_D
      end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
      local fevalG1 = function(x)
          collectgarbage()
          if x ~= parameters_G then -- get new parameters
              parameters_G:copy(x)
          end

          gradParameters_G:zero() -- reset gradients

          -- forward pass
          local samples = model_G:forward(LR_inputs)
          local g       = criterion_G:forward(samples[2], HR_inputs)
          local loss_tri = criterion_T:forward(samples[1])

          err_gen       = err_gen + g  
          local outputs = model_D:forward(samples[2])
          local f       = criterion_D:forward(outputs, targets)

          local samples_percep = preprocess_image(samples[2]:clone())
          local inputs_hr_percep = preprocess_image(HR_inputs:clone())  
          local err_percep = PerceptionLoss:forward(samples_percep, inputs_hr_percep)
          --io.write("G:",f+g, " G:", tostring(sgdState_G.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_G.numUpdates, " ", sgdState_D.numUpdates , "\n")
          --io.flush()

          --  backward pass
          local df_samples = criterion_D:backward(outputs, targets)
          model_D:backward(samples[2], df_samples) 
          local df_G_samples = criterion_G:backward(samples[2], HR_inputs)   ---added by xin
          local df_T = criterion_T:backward(samples[1])
          local df_vgg = PerceptionLoss:backward(samples_percep, inputs_hr_percep):div(127.5)   
          local df_do = model_D.modules[1].gradInput * opt.lambda + df_G_samples + df_vgg * opt.eta
          model_G:backward(LR_inputs, {df_T * 0, df_do})

  --      print('gradParameters_G', gradParameters_G:norm())
          return f,gradParameters_G
      end
	
	  local fevalG2 = function(x)
          collectgarbage()
          if x ~= parameters_G then -- get new parameters
              parameters_G:copy(x)
          end

          gradParameters_G:zero() -- reset gradients

          -- forward pass
          local samples = model_G:forward(LR_inputs)
          local g       = criterion_G:forward(samples[2], HR_inputs)
          local loss_tri = criterion_T:forward(samples[1])

          err_gen       = err_gen + g  
          local outputs = model_D:forward(samples[2])
          local f       = criterion_D:forward(outputs, targets)

          local samples_percep = preprocess_image(samples[2]:clone())
          local inputs_hr_percep = preprocess_image(HR_inputs:clone())  
          local err_percep = PerceptionLoss:forward(samples_percep, inputs_hr_percep)
          --io.write("G:",f+g, " G:", tostring(sgdState_G.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_G.numUpdates, " ", sgdState_D.numUpdates , "\n")
          --io.flush()

          --  backward pass
          local df_samples = criterion_D:backward(outputs, targets)
          model_D:backward(samples[2], df_samples) 
          local df_G_samples = criterion_G:backward(samples[2], HR_inputs)   ---added by xin
          local df_T = criterion_T:backward(samples[1])
          local df_vgg = PerceptionLoss:backward(samples_percep, inputs_hr_percep):div(127.5)   
          local df_do = model_D.modules[1].gradInput * opt.lambda + df_G_samples + df_vgg * opt.eta
          model_G:backward(LR_inputs, {df_T * opt.triloss, df_do * 0})

  --      print('gradParameters_G', gradParameters_G:norm())
          return f,gradParameters_G
      end

    ----------------------------------------------------------------------
      IDX_Batch = prepare_MiniBatch(IDX[{{t, t+opt.batchSize-1}}])
      --print(IDX_Batch)
      --print(IDX[{{t, t+opt.batchSize-1}}])
      local dataset_HR_samples = dataset_HR:index(IDX_Batch)  ------ Maybe right
      dataset_HR_samples:mul(2):add(-1)
      --print(dataset_HR_samples:size())
      dataset_HR_samples = dataset_HR_samples:index(2,torch.LongTensor{3,2,1})

      inputs[{{1, dataBatchSize}}] = dataset_HR_samples[{{1, dataBatchSize}}]:clone()
      
      local sample = torch.Tensor(dataBatchSize, opt.geometry[1], input_scale, input_scale)

      sample:copy(dataset_LR:index(1,IDX_Batch[{{1, dataBatchSize}}]))
      local tmp = model_G:forward(sample)
      inputs[{{dataBatchSize+1,opt.batchSize}}] = tmp[2]:clone()
      
      targets[{{1,dataBatchSize}}]:fill(1)	
      targets[{{dataBatchSize+1, opt.batchSize}}]:fill(0)

      rmsprop(fevalD, parameters_D, sgdState_D)


    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))

      HR_inputs:copy(dataset_HR_samples)
      LR_inputs:copy(dataset_LR:index(1, IDX_Batch))

      targets:fill(1)
      rmsprop(fevalG2, parameters_G, sgdState_G)
	  rmsprop(fevalG1, parameters_G, sgdState_G)  

       -- display progress
       --xlua.progress(t, N)
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / N
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

--  local err_val = adversarial.test(val_HR, valData_LR)
--  trainLogger:add{err_gen/(N/opt.batchSize), err_val }

  -- next epoch
  epoch = epoch + 1


  if epoch % opt.saveFreq == 0 then
      netname = string.format('adversarial_net_%s',epoch)
      local filename = paths.concat(opt.save, netname)    
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('<trainer> saving network to '..filename)
      model_D:clearState()
      model_G:clearState()  
      torch.save(filename, {G = model_G})
  end


end


-- test function
function adversarial.test(dataset_HR, dataset_LR_L1)
	
	model_G:evaluate()
	epoch = epoch or 0
	local N = 1024
	local time = sys.clock()
	local G_L16 = 0
  local batchSize = 64
	
	local inputs_HR = torch.Tensor(batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
	local inputs_LR1 = torch.Tensor(batchSize, opt.geometry[1], 16, 16)
	
	for t = 1,N,batchSize do
		inputs_HR = dataset_HR:index(torch.range(t, t+batchSize-1)):clone()
		inputs_HR = inputs_HR:index(2, torch.LongTensor{3,2,1}):cuda()
		inputs_HR:mul(2):add(-1)
				
		inputs_LR1:copy(dataset_LR_L1[{{t, t+batchSize-1}}])
		
		
		-- forward pass
		local samples = model_G:forward(inputs_LR1:cuda())
		local err_g   = criterion_G:forward(samples[2], inputs_HR:cuda())
		G_L16 = G_L16 + err_g
		
	end
	
	return G_L16/(N/batchSize)

end


return adversarial
