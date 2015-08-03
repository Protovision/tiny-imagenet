require "String"
require "File"

require "torch"
require "image"

TinyImageNet = function( sourceDir )
	local tin
	tin = {}

	sourceDir = sourceDir or "."
	tin._ = {}
	tin._.sourceDir = sourceDir
	tin._.imageDepth = 3
	tin._.imageWidth = 64
	tin._.imageHeight = 64
	tin._.numLabels = 200
	tin._.numTrainingSamplesPerLabel = 500
	tin._.numValidationSamples = 10000

	if File.exists( sourceDir .. "/classes.table" ) then
		tin._.classes = torch.load( sourceDir .. "/classes.table" )
	else
		local fs, line, i
		tin._.classes = {}
		tin._.classes.numberToWnid = {}
		tin._.classes.wnidToNumber = {}
		tin._.classes.wnidToWords = {}
		fs = io.open( sourceDir .. "/wnids.txt", "r" )
		if fs == nil then
			return nil
		end
		i = 1
		while true do
			line = fs:read( "*l" )
			if line == nil then
				break
			end
			table.insert( tin._.classes.numberToWnid, line )
			tin._.classes.wnidToNumber[ line ] = i
			i = i + 1
			tin._.classes.wnidToWords[ line ] = ""
		end
		fs:close()
		fs = io.open( sourceDir .. "/words.txt", "r" )
		if fs == nil then
			return nil
		end
		while true do
			local wnid
			line = fs:read( "*l" )
			if line == nil then
				break
			end
			wnid = String.token( line, " \t", 1 )
			if tin._.classes.wnidToWords[ wnid ] ~= nil then
				tin._.classes.wnidToWords[ wnid ] = string.sub(line, String.firstOf(line, String.token(line, " \t", 2)) )
			end
		end
		fs:close()
		torch.save( sourceDir .. "/classes.table", tin._.classes )
	end

	tin.getImageSize = function( self )
		return self._.imageDepth * self._.imageWidth * self._.imageHeight
	end

	tin.getNumberOfClasses = function( self )
		return self._.numLabels
	end

	tin.getWordsFromLabel = function( self, i )
		return self._.classes.wnidToWords[ self._.classes.numberToWnid[i] ]
	end
	tin.getWnidFromLabel = function( self, i )
		return self._.classes.numberToWnid[ i ]
	end

	tin.TrainingSet = function( self )
		local ts, trainPath, sampleIdx
		ts = {}
		ts._ = {}
		ts._.size = self._.numLabels * self._.numTrainingSamplesPerLabel

		trainPath = self._.sourceDir .. "/train"
		if File.exists( trainPath .. "/TrainingImages.Tensor" ) and 
		File.exists( trainPath .. "/TrainingLabels.Tensor" ) then
			ts.inputs  = torch.load( trainPath .. "/TrainingImages.Tensor" )
			ts.targets = torch.load( trainPath .. "/TrainingLabels.Tensor" )
		else
			ts.inputs = torch.Tensor(
				ts._.size,
				self._.imageDepth * self._.imageWidth * self._.imageHeight
			)
			ts.targets = torch.Tensor( ts._.size )
			sampleIdx = 1 
			for i = 1, #self._.classes.numberToWnid, 1 do
				local wnid, imagePathPrefix
				wnid = self._.classes.numberToWnid[i]
				imagePathPrefix = trainPath .. "/" .. wnid .. "/images/" .. wnid .. "_"
				for j = 0, self._.numTrainingSamplesPerLabel-1, 1 do
					local img
					img = image.load( imagePathPrefix .. j .. ".JPEG", 3 )
					if img == nil then
						return nil
					end
					ts.inputs[sampleIdx] = img:view(
						self._.imageDepth *
						self._.imageWidth *
						self._.imageHeight
					)
					ts.targets[sampleIdx] = i
					sampleIdx = sampleIdx + 1
				end
			end
			torch.save( trainPath .. "/TrainingImages.Tensor", ts.inputs )
			torch.save( trainPath .. "/TrainingLabels.Tensor", ts.targets )
		end

		ts.size = function( self )
			return self._.size
		end
	
		setmetatable( ts, {
			__index = function( self, idx )
				return {
					self.inputs[ idx ],
					self.targets[ idx ]
				}
			end
		} )

		return ts
	end

	tin.ValidationSet = function ( self )
		local vs, valPath
		vs = {}
		vs._ = {}
		vs._.size = self._.numValidationSamples

		valPath = self._.sourceDir .. "/val"
		if File.exists( valPath .. "/ValidationImages.Tensor" ) and
		File.exists( valPath .. "/ValidationLabels.Tensor" ) then
			vs.inputs = torch.load( valPath .. "/ValidationImages.Tensor" )
			vs.targets = torch.load( valPath .. "/ValidationLabels.Tensor" )
		else
			local fs, line, sampleIdx, imagePathPrefix
			vs.inputs = torch.Tensor(
				self._.numValidationSamples, 
				self._.imageDepth *
				self._.imageWidth *
				self._.imageHeight
			)
			vs.targets = torch.Tensor( self._.numValidationSamples )
			fs = io.open( self._.sourceDir .. "/val/val_annotations.txt", "r" )
			if fs == nil then
				return nil
			end
			sampleIdx = 1
			imagePathPrefix = self._.sourceDir .. "/val/images/val_"
			while true do
				local line, wnid, img
				line = fs:read( "*l" )
				if line == nil then
					break
				end
				wnid = String.token( line, " \t", 2 )
				img = image.load( imagePathPrefix .. (sampleIdx-1) .. ".JPEG", 3 )

				vs.inputs[ sampleIdx ] = img:view(
					self._.imageDepth *
					self._.imageWidth *
					self._.imageHeight
				)
				vs.targets[ sampleIdx ] = self._.classes.wnidToNumber[ wnid ]
				sampleIdx = sampleIdx + 1
			end
			fs:close()
			torch.save( valPath .. "/ValidationImages.Tensor", vs.inputs )
			torch.save( valPath .. "/ValidationLabels.Tensor", vs.targets )
		end

		vs.size = function( self )
			return self._.size
		end

		setmetatable( vs, {
			__index = function( self, idx )
				return {
					self.inputs[ idx ],
					self.targets[ idx ]
				}
			end
		} )

		return vs
	end

	return tin
end
