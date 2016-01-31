--[[---------------------------------------------------------------------------

	Module:		Array

	Author:		Protovision

	Date:		August 1, 2015

	Description:	Functions for operating on lua arrays.
			Will expand in the future.

-----------------------------------------------------------------------------]]


Array = {}

Array.firstOf = function( table, value )
	for i = 1, #table, 1 do
		if table[i] == value then
			return i
		end
	end
	return nil
end

Array.range = function( min, max )
	local array
	array = {}
	for i = min, max, 1 do
		table.insert( array, i )
	end
	return array
end

