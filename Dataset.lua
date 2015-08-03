--[[---------------------------------------------------------------------------

	Module:		Dataset

	Dependencies:	torch, Array

	Author:		Mark Swoope

	Date:		August 1, 2015

	Description:	Collection of functions to operate on datasets.
			A dataset is a collection of samples. A dataset 
			is expected to have the following methods:

			size():
			Returns the number of samples in the dataset

			[ i ]:
			Returns a 2-element array for the sample at index
			i within the dataset. The first element should be 
			a vector of input values and the second element 
			should be the target value.

			A dataset may optionally contain these attributes 
			for faster batch processing:

			inputs:
			A 2-dimensional matrix where each row contains 
			the input values for a sample.

			targets:
			A vector where each element is the target value 
			for a sample.

			A "light dataset" does not contain the inputs 
			or targets attributes. A "full dataset" does
			contain the inputs and targets attributes.

	Functions:	Dataset.shuffle( dataset, maxSize, makeCopy )

			Takes maxSize random samples out of a dataset. 
			If maxSize is nil, then maxSize defaults to 
			dataset:size().
			If makeCopy is true, this function will return 
			a full dataset, otherwise it will return a 
			light dataset by default.

			Dataset.selectTargets( dataset, allowedTargets, 
				makeCopy )

			Gets samples out of dataset whose target value is 
			contained within allowedTargets. 
			If makeCopy is true, this function will return a
			full dataset, otherwise it will return a light 
			dataset by default.

			Dataset.process( dataset, batchSize, callback )

			Calls the subprocedure callback for every batchSize 
			samples from the dataset. callback should expect 
			two arguments: the 2-dimensional tensor of inputs, 
			and the vector of target values. 

-----------------------------------------------------------------------------]]

require "torch"
require "Array"

Dataset = {}
Dataset._ = {}

Dataset._.shuffleWithoutCopy = function( dataset, maxSize )
	local sd, shuffledIndices

	sd = {}
	sd._ = {}
	sd._.dataset = dataset
	sd._.indices = {}
	shuffledIndices = torch.randperm( dataset:size() )
	maxSize = maxSize or dataset:size()
	for i = 1, maxSize, 1 do
		table.insert( sd._.indices, shuffledIndices[i] )
	end

	sd.size = function( self )
		return #self._.indices
	end

	setmetatable( sd, {
		__index = function( self, idx )
			return self._.dataset[ self._.indices[idx] ]
		end
	} )
	return sd
end

Dataset._.shuffleWithCopy = function( dataset, maxSize )
	local sd, shallowShuffledDs

	sd = {}
	shallowShuffledDs = Dataset._.shuffleWithoutCopy( dataset, maxSize )
	sd._.size = shallowShuffledDs:size()
	sd.inputs = torch.Tensor( sd._.size, shallowShuffledDs[1][1]:size(1) )
	sd.targets = torch.Tensor( sd._.size )
	for i = 1, sd._.size, 1 do
		local sample
		sample = shallowShuffledDs[i]
		sd.inputs[i] = sample[1]
		sd.targets[i] = sample[2]
	end

	sd.size = function( self )
		return self._.size
	end

	setmetatable( ds, {
		__index = function( self, idx )
			return { self.inputs[idx], self.targets[idx] }
		end
	} )
	return sd
end

Dataset._.selectTargetsWithoutCopy = function( dataset, allowedTargets )
	local ds

	ds = {}
	ds._ = {}
	ds._.dataset = dataset
	ds._.indices = {}
	for i = 1, dataset:size(), 1 do
		local sample
		sample = dataset[i]
		if Array.firstOf( allowedTargets, sample[2] ) ~= nil then
			table.insert( ds._.indices, i )
		end
	end

	ds.size = function( self )
		return #self._.indices
	end

	setmetatable( ds, {
		__index = function( self, idx ) 
			return self._.dataset[ self._.indices[idx] ]
		end
	} )
	return ds
end

Dataset._.selectTargetsWithCopy = function( dataset, allowedTargets )
	local ds, shallowDs

	ds = {}
	ds._ = {}
	shallowDs = Dataset._.selectTargetsWithoutCopy( dataset,
		allowedTargets )
	ds._.size = shallowDs:size()
	ds.inputs = torch.Tensor( ds._.size, shallowDs[1][1]:size(1) )
	ds.targets = torch.Tensor( ds._.size )

	for i = 1, ds._.size, 1 do
		local sample
		sample = shallowDs[i]
		ds.inputs[i] = sample[1]
		ds.targets[i] = sample[2]
	end

	ds.size = function( self )
		return self._.size
	end

	setmetatable( ds, {
		__index = function( self, idx )
			return { self.inputs[idx], self.targets[idx] }
		end
	} )
	return ds
end

Dataset._.slowProcess = function( dataset, batchSize, callback )
	local samplesProcessed

	samplesProcessed = 0
	while true do
		local sample, inputBatch, targetBatch, samplesRemaining, size

		collectgarbage()
		samplesRemaining = dataset:size() - samplesProcessed
		if samplesRemaining >= batchSize then
			size = batchSize
		else
			size = samplesRemaining
		end

		inputBatch = torch.Tensor( size, dataset[1][1]:size(1) )
		targetBatch = torch.Tensor( size )

		for k = 1, size, 1 do
			sample = dataset[ samplesProcessed+k ]
			inputBatch[k] = sample[1]
			targetBatch[k] = sample[2]
		end

		callback( inputBatch, targetBatch )

		samplesProcessed = samplesProcessed + size
		if samplesProcessed >= dataset:size() then
			break
		end
	end
end

Dataset._.fastProcess = function( dataset, batchSize, callback )
	local samplesProcessed

	samplesProcessed = 0
	while true do
		local inputBatch, targetBatch, samplesRemaining, size

		collectgarbage()
		samplesRemaining = dataset:size() - samplesProcessed
		if samplesRemaining >= batchSize then
			size = batchSize
		else
			size = samplesRemaining
		end

		inputBatch = dataset.inputs:sub( samplesProcessed+1,
			samplesProcessed+size, 1, dataset.inputs:size(2) )
		targetBatch = dataset.targets:sub( samplesProcessed+1,
			samplesProcessed+size )

		callback( inputBatch, targetBatch )

		samplesProcessed = samplesProcessed + size
		if samplesProcessed >= dataset:size() then
			break
		end
	end
end

Dataset.process = function( dataset, batchSize, callback )
	if rawget(dataset, "inputs") == nil or rawget(dataset, "targets") == nil then
		return Dataset._.slowProcess( dataset, batchSize, callback )
	else
		return Dataset._.fastProcess( dataset, batchSize, callback )
	end
end

Dataset.selectTargets = function( dataset, allowedTargets, makeCopy )
	if makeCopy == true then
		return Dataset._.selectTargetsWithCopy( dataset,
			allowedTargets )
	else
		return Dataset._.selectTargetsWithoutCopy( dataset,
			allowedTargets )
	end
end

Dataset.shuffle = function( dataset, maxSize, makeCopy )
	if maxSize == nil or maxSize <= 0 then
		maxSize = dataset:size()
	end

	if makeCopy == true then
		return Dataset._.shuffleWithCopy( dataset, maxSize )
	else
		return Dataset._.shuffleWithoutCopy( dataset, maxSize )
	end
end


