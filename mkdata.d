import std.algorithm.iteration;
import std.array;
import std.conv;
import std.exception;
import std.file;
import std.format;
import std.parallelism;
import std.path;
import std.random;
import std.range;
import std.stdio;
import std.string;
import std.traits;

import ae.utils.array;
import ae.utils.funopt;
import ae.utils.main;

import metadata;
import data;
import tests;

shared string fnBase;
uint numTests;

bool success;

struct DataFile
{
	string name;
	DataBuffer[] buffers;
	Format format;

	this(string name, Format format)
	{
		this.name = name;
		enforce(format, "No format specified for " ~ name);
		this.format = format;

		this.buffers.length = numTests;
		foreach (ref buf; this.buffers)
			buf.format = format;
	}

	@disable this(this);

	~this()
	{
		if (this.name && success)
		{
			auto fn = fnBase ~ "-" ~ name ~ "." ~ text(format);
			stderr.writeln("Saving " ~ fn);
			auto fnTmp = fn ~ ".tmp";
			auto f = File(fnTmp, "wb");

			foreach (ref buf; buffers)
				f.rawWrite(buf.buf.peek);

			f.close();
			rename(fnTmp, fn); // strip .tmp
		}
	}
}

void mkdata(
	Option!(uint, "Block width (default = 0 = unlimited)") blockWidth,
	Option!(uint, "Block height (default = 0 = unlimited)") blockHeight,
	Option!(Preprocessing, "Preprocessing to apply to pixel data (" ~ [EnumMembers!Preprocessing].map!text.join("|") ~ ")") pre,
	Option!(Format, "Pixel format (" ~ [EnumMembers!Format].map!text.join("|") ~ ")") pixelFormat,
	Option!(Format, "Label format (" ~ [EnumMembers!Format].map!text.join("|") ~ ")") labelFormat,
	Option!(uint, "Size (width and height) of area to sample (0 = use entire image)") imageSize,
	Option!(Order, "Order of samples (" ~ [EnumMembers!Order].map!text.join("|") ~ ")") sampleOrder,
	Option!(uint, "Sample limit per image (0 = use all samples)") imageSamples,
	Option!(float, "Sample fraction per image (instead of image-samples)") imageSamplesFraction,
	Option!(bool, "Whether to filter \"empty\" samples") filterSamples,
	Option!(uint, "Random seed") seed = 0,
	Option!(uint, "Number of tests to convert (default = all)") numTests = 0,
)
{
	if (!numTests)
		numTests = tests.numTests;
	.numTests = numTests;

	fnBase = [
		"data",
		format!"%dx%d"(blockWidth.value, blockHeight.value),
		format!"size_%s"(imageSize.value),
		imageSamplesFraction.value !is float.init
	?	format!"samplesfraction_%s"(imageSamplesFraction.value)
	:	format!"samples_%s"(imageSamples.value),
		format!"order_%s"(sampleOrder.value),
		format!"filter_%s"(filterSamples.value),
		format!"tests_%s"(numTests.value),
	].join("-");
	stderr.writeln("Data filename base: ", fnBase);

	size_t numSamples;
	scope(success) stderr.writefln("%s: Wrote %d samples.", fnBase, numSamples);

	auto pixels = DataFile("pixels-" ~ text(pre.value), pixelFormat);
	auto labels = DataFile("labels", labelFormat);
	auto metadata = DataFile("metadata", Format.f32);
	auto indices = DataFile("indices", Format.u32);

	ImageProcessor processor;
	processor.blockWidth           = blockWidth          ;
	processor.blockHeight          = blockHeight         ;
	processor.pre                  = pre                 ;
	processor.imageSize            = imageSize           ;
	processor.sampleOrder          = sampleOrder         ;
	processor.imageSamples         = imageSamples        ;
	processor.imageSamplesFraction = imageSamplesFraction;
	processor.filterSamples        = filterSamples       ;

	void loadTestImage(in ref Test test, bool which, uint index)
	{
		auto fileName = test.getFileName(which);

		auto numImageSamples = processor.loadImage(
			test.images[which],
			which, fileName.getSize, index,
			test.info.images[which].format, test.info.images[which].quality, 
			pixels  .buffers[index],
			labels  .buffers[index],
			metadata.buffers[index],
			indices .buffers[index],
			null,
		);
		synchronized numSamples += numImageSamples;
	}

	void loadTest(uint index)
	{
		rndGen.seed(seed * numTests + index);
		stderr.writef("%d/%d\r", index, numTests); stderr.flush();
		auto test = getTest(index);
		foreach (which; [false, true])
			loadTestImage(test, which, index);
	}

	void loadData()
	{
		uint index;
		foreach (_; numTests.iota.parallel(1))
		{
			uint i;
			synchronized i = index++;
			loadTest(i);
		}
	}

	loadData();

	success = true;
}

mixin main!(funopt!mkdata);
