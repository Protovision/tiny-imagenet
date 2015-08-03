require "nn"

--[[
Model2 = function()
	local model
	model = nn.Sequential()
	model:add( nn.Reshape(3,64,64) )
	model:add( nn.SpatialConvolutionMM(3,64,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.SpatialConvolutionMM(64,128,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.SpatialConvolutionMM(128,512,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.Reshape(512*8*8) )
	model:add( nn.Linear(512*8*8, 2) )
	model:add( nn.LogSoftMax() )
	return model
end
--]]

Model = function()
	local model
	model = nn.Sequential()
	model:add( nn.Reshape(3,64,64) )
	model:add( nn.SpatialConvolutionMM(3,64,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.SpatialConvolutionMM(64,128,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.SpatialConvolutionMM(128,256,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.SpatialConvolutionMM(256,512,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1) )
	model:add( nn.ReLU(true) )
	model:add( nn.SpatialMaxPooling(2,2,2,2) )
	model:add( nn.Reshape(512*4*4) )
	model:add( nn.Dropout(0.5) )
	model:add( nn.Linear(512*4*4, 4096) )
	model:add( nn.ReLU(true) )
	model:add( nn.Dropout(0.5) )
	model:add( nn.Linear(4096, 4096) )
	model:add( nn.ReLU(true) )
	model:add( nn.Linear(4096, 200) )
	model:add( nn.LogSoftMax() )
	return model
end
