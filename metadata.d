import std.algorithm.searching;
import std.conv;
import std.file;
import std.stdio;
import std.string;

import ae.sys.cmd;
import ae.utils.array;
import ae.utils.meta;

import data;

FileFormat getFileFormat(string fileName)
{
	auto header = fileName.read(4).bytes;
	if (header.startsWith([ubyte(0xFF), ubyte(0xD8)]))
		return FileFormat.JPEG;
	if (header == "\x89PNG".representation)
		return FileFormat.PNG;
	if (header == "GIF8".representation)
		return FileFormat.GIF;
	stderr.writeln("Unknown file format: ", fileName);
	return enumLength!FileFormat; // neither
}

float getQuality(string fileName)
{
	try
		return query(["identify", "-format", "%Q", fileName ~ "[0]"]).strip().to!float / 100f;
	catch (Exception e)
	{
		stderr.writeln(fileName, ": ", e);
		return 0;
	}
}
