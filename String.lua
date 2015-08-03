--[[--------------------------------------------------------------------------

	Module:		String

	Author:		Mark Swoope

	Date:		August 1, 2015

	Description:	Functions for manipulating strings

	Functions:	String.token( str, delim, i )

			Extracts the ith token from the string str.
			Tokens are delimited by the characters in string 
			delim.

			String.split( str, delim )

			Returns an array of string tokens from the string 
			str. The tokens are delimited by the characters in 
			string delim.

			String.firstOf( str, substr )

			Returns the index in string str for the first 
			occurance of substring substr.

----------------------------------------------------------------------------]]

String = {}

String.token = function( str, delim, idx )
	local i
	i = 1
	for w in string.gmatch(str, "([^" .. delim .. "]+)") do
		if i == idx then
			return w
		end
		i = i + 1
	end
	return nil
end

String.split = function( str, delim )
	local tokens
	tokens = {}
	for w in string.gmatch(str, "([^" .. delim .. "]+)") do
		table.insert( tokens, w )
	end
	return tokens
end

String.firstOf = function( str, substr )
	return string.find( str, substr, 1, true )
end


