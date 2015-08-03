--[[---------------------------------------------------------------------------

	Module:		Classifier

	Dependencies:	cutorch, cunn, optim, Dataset

	Author:		Mark Swoope

	Date:		August 1, 2015

	Description:	Uses a neural network to perform classification.

	Constructor:	Classifier()
			Creates the classifier with the following default 
			configurations:
				Learning rate: 0.005
				Training batch size: 10
				Testing batch size: 1000
				Learning rate decay: 0
				Weight decay: 0
				Momentum: 0
				Criterion: Negative Log Likelihood

			You must set a neural network model using setModel before 
			training, testing, or classifying data.

			Classifier( saveFile )
			Loads a classifier with it's underlying neural network from 
			a file

	Attributes:	learningRate
		
			learningRateDecay

			weightDecay

			momentum

			trainBatchSize

			testBatchSize

	Methods:	getModel()

			getCriterion()

			getParameters()

			getGradParameters()

			setCUDA( boolean )

			Transfers the underlying neural network model to the 
			GPU. Training and testing will now be performed on the
			GPU. Set to false to move back to CPU.

			setModel( model )

			setCriterion( criterion )

			backupParameters()

			Saves the parameters to the underlying neural network

			restoreParameters()

			Restores the parameters to the underlying neural network 
			from the last call to backupParameters()

			classify( input )

			Returns an integer representing the classification 
			of the input as calculated by the underlying network.

			train( dataset )

			Performs 1 epoch of training over the dataset.
			Returns a table containing the keys: time and loss.
			time represents the number of seconds that the train 
			method ran for, loss represents the the loss of the 
			neural network after the train method completed.

			test( dataset )

			Tests the classifier against the dataset. Returns 
			a table containing the keys: time, accuracy, and 
			loss. time represents the number of seconds that the
			test method ran for, loss represents the loss of the 
			neural network compared to the dataset, accuracy 
			represents the percentage of samples from the dataset
			for which the neural network correctly classified.

			save( saveFile )
			Saves the classifier and it's underlying neural network 
			to a file.

-----------------------------------------------------------------------------]]
require "Dataset"

require "cutorch"
require "cunn"
require "optim"

Classifier = function( saveFile )
	local args

	if saveFile ~= nil then
		if File.exists( saveFile ) then
			return torch.load( saveFile )
		else
			return nil
		end
	end

	local c
	c = {}

	c._ = {}
	c._.useCUDA = false
	c._.criterion = nn.ClassNLLCriterion()
	c.trainBatchSize = 10
	c.testBatchSize = 1000
	c.learningRate = 0.005
	c.weightDecay = 0
	c.momentum = 0
	c.learningRateDecay = 0

	c._.Input = function( self, input )
		if self.useCUDA then
			return input:cuda()
		else
			return input
		end
	end

	c._.Target = function( self, output )
		if self.useCUDA then
			return torch.CudaTensor( {output} )
		else
			return output
		end
	end

	c._.InputBatch = function( self, inputBatch )
		if self.useCUDA then
			return inputBatch:cuda()
		else
			return inputBatch
		end
	end

	c._.TargetBatch = function( self, targetBatch )
		if self.useCUDA then
			return targetBatch:cuda()
		else
			return targetBatch
		end
	end

	c.backupParameters = function( self )
		self._.savedParameters = self._.parameters:clone()
	end

	c.restoreParameters = function( self )
		self._.parameters:copy( self._.savedParameters )
	end

	c.save = function( self, file )
		torch.save( file, self )
	end

	c.getModel = function( self )
		if self._.useCUDA then
			return self._.cudaModel
		else
			return self._.model
		end
	end

	c.getCriterion = function( self )
		if self._.useCUDA then
			return self._.cudaCriterion
		else
			return self._.criterion
		end
	end

	c.getParameters = function( self )
		return self._.parameters
	end

	c.getGradParameters = function( self )
		return self._.gradParameters
	end

	c.setModel = function( self, model )
		self._.model = model
		self._.parameters, self._.gradParameters = model:getParameters()
	end

	c.setCriterion = function( self, criterion )
		self._.criterion = criterion
	end

	c.setCUDA = function( self, bool )
		self._.useCUDA = bool
		if bool == true then
			self._.cudaCriterion = self._.criterion:cuda()
			self._.cudaModel = self._.model:cuda()
			self._.parameters, self._.gradParameters = self._.cudaModel:getParameters()
		else
			self._.paramters, self._.gradParameters = self._.model:getParameters()
		end
	end

	c.classify = function( self, data )
		local input, output, model, classification

		model = self:getModel()
		model:evaluate()
		input = self._:Input( data )
		output = model:forward( input )
		_, classification = torch.max( output, 1 )
		return classification:squeeze()
	end

	c.train = function( self, trainingSet )
		local model, criterion, timer, shuffledTrainingSet, trainingLoss, realBatchSize

		timer = torch.Timer()
		model = self:getModel()
		model:training()
		criterion = self:getCriterion()
		shuffledTrainingSet = Dataset.shuffle( trainingSet )

		self._.sgdState = self._.sgdState or {
			learningRate = self.learningRate,
			learningRateDecay = self.learningRateDecay,
			weightDecay = self.weightDecay,
			momentum = self.momentum
		}
		trainingLoss = 0
		if self.trainBatchSize > trainingSet:size() then
			realBatchSize = trainingSet:size()
		else
			realBatchSize = self.trainBatchSize
		end
		Dataset.process( shuffledTrainingSet, realBatchSize,
			function( inputBatch, targetBatch )
				local feval
				feval = function( x )
						local inputs, targets, f, df_do, predictions, gradients
						inputs = self._:InputBatch( inputBatch )
						targets = self._:TargetBatch( targetBatch )
						if x ~= self._.parameters then
							self._.parameters:copy( x )
						end
						self._.gradParameters:zero()
						predictions = model:forward( inputs )
						f = criterion:forward( predictions, targets )
						trainingLoss = trainingLoss + f
						df_do = criterion:backward( predictions, targets )
						model:backward( inputs, df_do )
						return f, self._.gradParameters
					end
				optim.sgd( feval, self._.parameters, self._.sgdState )

			end
		)
		elapsed = timer:time().real
		return {
			time = elapsed,
			loss = trainingLoss * realBatchSize / trainingSet:size()
		}
	end

	c.test = function( self, testSet )
		local model, criterion, timer, elapsed, testErrors, realBatchSize

		timer = torch.Timer()
		model = self:getModel()
		model:evaluate()
		criterion = self:getCriterion()
  
                totalErrors = 0
		loss = 0
		if self.testBatchSize > testSet:size() then
			realBatchSize = testSet:size()
		else
			realBatchSize = self.testBatchSize
		end

		Dataset.process( testSet, realBatchSize, 
			function( inputBatch, targetBatch )
				inputs = self._:InputBatch( inputBatch )
				targets = self._:TargetBatch( targetBatch )
				predictions = model:forward( inputs )
				loss = loss + criterion:forward( predictions, targets )
				for i = 1, predictions:size(1), 1 do
					local output, _, classification
					output = predictions[i]
					_, classification = torch.max( output, 1 )
					if classification[1] ~= targets[i] then
						totalErrors = totalErrors + 1
					end
				end
			end
		)
		elapsed = timer:time().real
		return {
			time = elapsed,
                        accuracy = (testSet:size() - totalErrors)/testSet:size() * 100,
			loss = loss * realBatchSize / testSet:size()
		}
	end

	return c
end
