import std.file;
import std.format;
import std.parallelism;
import std.range;
import std.stdio;

import ae.sys.data;
import ae.sys.datamm;
import ae.utils.graphics.color;
import ae.utils.graphics.im_convert;
import ae.utils.graphics.image;
import ae.utils.json;

import data : FileFormat;

alias TestImage = typeof(viewBMP!BGR((const(void)[]).init));

struct TestInfo
{
	struct Image
	{
		string fileName;
		uint width, height;
		ulong fileSize;
		FileFormat format;
		float quality;
		uint colorCount;
	}
	Image[2] images; // orig, edit
	string[] imCmdLine;
}

struct Test
{
	TestImage[2] images;
	Data[2] data;
	TestInfo info;

	string getFileName(ubyte which) const { return info.images[which].fileName; }
}

enum numTests = 10_000;

Test getTest(int n)
{
	Test test;
	foreach (i, kind; ["orig", "edit"])
	{
		auto fn = "tests/%04d-%s.bmp".format(n, kind);
		test.data[i] = mapFile(fn, MmMode.read);
		test.images[i] = viewBMP!BGR(test.data[i].contents);
	}
	test.info = readText("tests/%04d-info.json".format(n)).jsonParse!TestInfo;
	return test;
}
