--[[--------------------------------------------------------------------------

	Module:		File

	Dependencies:	serial, System

	Author:		Protovision

	Date:		August 1, 2015

	Description:	Functions related to the file system

	Functions:	File.exists( path )

			Returns true if the file path exists

			File.mkdir( path )

			Creates a directory and returns true 
			if the operation was successful. 

			File.rmdir( path )

			Removes a directory even if it's not empty.

			File.remove( path )

			Remove a file that is not a directory

			File.rename( oldPath, newPath )

			Renames a file

			File.open( path, mode )

			Same as io.open

			File.openBinary( path, mode, endian )

			Opens the file path for tranferring binary data.
			mode is an fopen style open file mode string. 
			endian can be "le" for little endian or "be" for 
			big endian. 
			
			The returned file stream has the following methods:

			read( typename ):
			Reads from the file. typename specifies the format 
			of data to read. typename can be: "uint8", "sint8",
			"uint16", "sint16", "uint32", "sint32", "uint64",
			"sint64", "float," "double", "char", "cstring"

			write( value, typename ):
			Writes a value to the file. See read method for 
			available typename formats.

			close():
			Closes the file


----------------------------------------------------------------------------]]

require "System"

File = {}
File._ = {}
File._.serial = require "serial"
File._.platform = System.platform()

File.exists = function( path )
	local fs
	fs = io.open( path, "r" )
	if fs == nil then
		return false
	end
	fs:close()
	return true
end

File.mkdir = function( path )
	local a,b,c
	a,b,c = os.execute( "mkdir \"" .. path .. "\"" )
	return c == 0
end

File.rmdir = function( path )
	local a,b,c
	if platform == "Microsoft" then
		a,b,c = os.execute( "rmdir /s /q \"" .. path .. "\"" )
	else
		a,b,c = os.execute( "rm -rf \"" .. path .. "\"" )
	end
	return c == 0
end

File.remove  = function( path )
	local a,b,c
	if platform == "Microsoft" then
		a,b,c = os.execute( "del /f /q \"" .. path .. "\"" )
	else
		a,b,c = os.execute( "rm -f \"" .. path .. "\"" )
	end
	return c == 0
end

File.rename = function( oldPath, newPath )
	local a,b,c
	if platform == "Microsoft" then
		a,b,c = os.execute( "rename \"" .. oldPath .. "\" \"" .. newPath .. "\"" )
	else
		a,b,c = os.execute( "mv \"" .. oldPath .. "\" \"" .. newPath .. "\"" )
	end
	return c == 0
end

File.open = function( path, mode )
	return io.open( path, mode )
end

File.openBinary = function( path, mode, endian )
	local f
	f = {}
	f._ = {}
	f._.endian = endian
	f._.fh = io.open( path, mode .. "b" )
	if f._.fh == nil then
		return nil
	end
	f._.fs = File._.serial.filestream( f._.fh )

	f.read = function( self, typename )
		return File._.serial.read[ typename ]( self._.fs, self._.endian )
	end

	f.write = function( self, value, typename )
		File._.serial.write[ typename ]( self._.fs, value, self._.endian )
	end

	f.close = function( self )
		self._.fh:close()
	end

	return f
end

