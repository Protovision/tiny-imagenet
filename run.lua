#!/usr/bin/env th

require "Classifier"
require "TinyImageNet"
require "models"

infinity = 1/0

cmd = torch.CmdLine()
cmd:text( "Tiny ImageNet Classifier Experiment" )
cmd:text( "Options: " )
cmd:option( "-load", false, "Load classifier and results from last run" )
cmd:option( "-maxEpochs", 200, "Number of epochs to perform" )
cmd:option( "-maxSeconds", infinity, "Stop before the next epoch if experiment runs over maxSeconds" )
cmd:option( "-saveResults", true, "Enable saving results" )
cmd:option( "-saveClassifier", true, "Enable saving classifier" )
cmd:option( "-epochsPerSave", 10, "Epochs to run before saving classifier and results" )
cmd:option( "-resultsFile", "results.table", "File to save results to" )
cmd:option( "-classifierFile", "classifier.table", "File to save classifier to" )
cmd:option( "-cuda", true, "Enable CUDA acceleration" )
cmd:option( "-batchSize", 30, "Batch size" )
cmd:option( "-learningRate", 0.01, "Learning rate" )
cmd:option( "-learningRateDecay", 0, "Learning rate decay" )
cmd:option( "-momentum", 0, "Momentum" )
cmd:option( "-weightDecay", 0.00005, "Weight decay" )
params = cmd:parse( arg )

if params.load == true then
	print( "Loading classifier..." )
	classifier = Classifier( params.classifierFile )
	results = torch.load( params.resultsFile )
else
	print( "Initializing classifier..." )
	classifier = Classifier()
	classifier:setModel( Model() )
	results = {}
	results.setup = {
		batchSize = params.batchSize,
		learningRate = params.learningRate,
		learningRateDecay = params.learningRateDecay,
		momentum = params.momentum,
		weightDecay = params.weightDecay
	}
	results.summary = {
		totalTrainTime = 0,
		totalTrainTestTime = 0,
		totalValTestTime = 0
	}
	results.epochs = {}
end
classifier:setCriterion( nn.ClassNLLCriterion() )
classifier.trainBatchSize = params.batchSize
classifier.learningRate = params.learningRate
classifier.learningRateDecay = params.learningRateDecay
classifier.momentum = params.momentum
classifier.weightDecay = params.weightDecay
classifier:setCUDA( params.cuda )

print( "Loading labels..." )
tinyImageNet = TinyImageNet( "tiny-imagenet-200" )
print( "Loading training set..." )
trainingSet = tinyImageNet:TrainingSet()
print( "Loading validation set..." )
validationSet = tinyImageNet:ValidationSet()

print( "Running epochs..." )
timer = torch.Timer()
io.write( string.format(
	"%-11s%-11s%-11s%-11s%-11s%-11s%-11s\n",
	"Epoch",
	"TrainTime",
	"ValTime",
	"TrainAcc",
	"ValAcc",
	"TrainLoss",
	"ValLoss"
) )

for epoch = #results.epochs + 1, params.maxEpochs, 1 do
	local train, trainTest, valTest
	if timer:time().real >= params.maxSeconds then
		break
	end
	train = classifier:train( trainingSet )
	trainTest = classifier:test( trainingSet )
	valTest = classifier:test( validationSet )
	io.write( string.format(
		"%-11d%-11.4f%-11.4f%-11.4f%-11.4f%-11.6f%-11.6f\n",
		epoch,
		train.time,
		trainTest.time + valTest.time,
		trainTest.accuracy,
		valTest.accuracy,
		trainTest.loss,
		valTest.loss
	) )
	table.insert( results.epochs, { train = train, trainTest = trainTest, valTest = valTest } )
	results.summary.totalTrainTime = results.summary.totalTrainTime + train.time
	results.summary.totalTrainTestTime = results.summary.totalTrainTestTime + trainTest.time
	results.summary.totalValTestTime = results.summary.totalValTestTime + valTest.time
	if (epoch % params.epochsPerSave == 0) or (timer:time().real >= params.maxSeconds) then
		if params.saveResults == true then
			print( "Saving results..." )
			results.summary.totalTime = timer:time().real
			results.summary.avgTrainTime = results.summary.totalTrainTime / epoch
			results.summary.avgTrainTestTime = results.summary.totalTrainTestTime / epoch
			results.summary.avgValTestTime = results.summary.totalValTestTime / epoch
			results.summary.finalTrainAcc = trainTest.accuracy
			results.summary.finalValAcc = valTest.accuracy
			results.summary.finalTrainLoss = trainTest.loss
			results.summary.finalValLoss = valTest.loss
			torch.save( params.resultsFile, results )
			print( "Results saved." )
		end
		if params.saveClassifier == true then
			print( "Saving classifier..." )
			classifier:save( params.classifierFile )
			print( "Classifier saved." )
		end
		if timer:time().real >= params.maxSeconds then
			break
		end
	end
end


