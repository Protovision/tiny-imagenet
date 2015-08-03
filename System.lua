
System = {}

System.platform = function( self )
	local p, line
	p = io.popen( "uname", "r" )
	if p == nil then
		return "Microsoft"
	end
	line = p:read( "*l" )
	p:close()
	if line == "Darwin" then
		return "Apple"
	else
		return "UNIX"
	end
end
