import std.algorithm.iteration;
import std.exception;
import std.format;
import std.process;
import std.range;
import std.stdio;

import ae.sys.cmd;
import ae.utils.funopt;
import ae.utils.graphics.bitmap;
import ae.utils.graphics.color;
import ae.utils.graphics.image;
import ae.utils.main;

import metadata;
import data;

enum ExecutionMode
{
	gpu,
	cpuMultiThreaded,
	cpuSingleThreaded,
}

struct Server
{
	ImageProcessor processor;
	File pythonInput, pythonOutput;
	Pid python;

	this(/*string modelFn, */ExecutionMode executionMode, bool filterSamples = true)
	{
		processor.blockWidth        = 8;
		processor.blockHeight       = 8;
		processor.imageSize         = 0;
		processor.sampleOrder       = Order.entropy;
		processor.imageSamples      = 512;
		processor.filterSamples     = filterSamples;

		string[string] environment;
		final switch (executionMode)
		{
			case ExecutionMode.gpu:
				break;
			case ExecutionMode.cpuMultiThreaded:
				environment["CUDA_VISIBLE_DEVICES"] = "-1";
				break;
			case ExecutionMode.cpuSingleThreaded:
				// https://github.com/tensorflow/tensorflow/issues/29968#issuecomment-789621000
				environment["OMP_NUM_THREADS"] = "1";
				environment["TF_NUM_INTRAOP_THREADS"] = "1";
				environment["TF_NUM_INTEROP_THREADS"] = "1";
				goto case ExecutionMode.cpuMultiThreaded;
		}

		auto pythonStdin = pipe();
		auto pythonStdout = pipe();
		python = spawnProcess(
			["./quality.sh", "server"/*, modelFn*/],
			pythonStdin.readEnd,
			pythonStdout.writeEnd,
			stderr,
			environment,
			Config.newEnv,
		);
		pythonInput = pythonStdin.writeEnd;
		pythonOutput = pythonStdout.readEnd;
	}

	struct ImageResult
	{
		float score;
	}

	ImageResult analyzeImage(V)(V image, ulong fileSize, FileFormat format, float quality)
	{
		DataBuffer pixels, dct, labels, metadata, indices, entropies, nul;
		pixels   .format = Format.u8;
		dct      .format = Format.f32;
		labels   .format = Format.u8;
		metadata .format = Format.f32;
		indices  .format = Format.u32;
		entropies.format = Format.u32;
		nul      .format = Format.nul;

		processor.pre = Preprocessing.none;

		processor.loadImage(
			image,
			false, fileSize, 0,
			format, quality,
			pixels  ,
			nul     ,
			nul     ,
			nul     ,
			&entropies,
		);

		processor.pre = Preprocessing.dct;

		uint numImageSamples = processor.loadImage(
			image,
			false, fileSize, 0,
			format, quality,
			dct     ,
			labels  ,
			metadata,
			indices ,
			null,
		);

		pythonInput.rawWrite((&numImageSamples)[0..1]);
		pythonInput.rawWrite(pixels  .buf.peek);
		pythonInput.rawWrite(dct     .buf.peek);
		pythonInput.rawWrite(metadata.buf.peek);
		pythonInput.flush();

		ImageResult[1] result;
		auto buf = result[];
		buf = pythonOutput.rawRead(buf);
		enforce(buf.length == result.length,
			"Did not read the entire result");

		return result[0];
	}

	static Image!BGR readImage(string fn)
	{
		return pipe([
			"convert",
			"-background", "white",
			"-alpha", "remove",
			"-type", "TrueColor",
			fn ~ "[0]",
			"bmp:-",
		], null).parseBMP!BGR();
	}

	ImageResult analyzeImage(string fn)
	{
		import std.file : read, getSize;
		return analyzeImage(
			readImage(fn),
			getSize(fn),
			getFileFormat(fn),
			getQuality(fn),
		);
	}

	@disable this(this);

	~this()
	{
		pythonInput.close();
		pythonOutput.close();
		enforce(wait(python) == 0, "python failed");
	}
}

/*
void server(string modelFn)
{

	alias Header = BitmapHeader!3;
	ubyte[] imageBuf;
	imageBuf.length = Header.sizeof;

	Image!BGR image;

	while (!stdin.eof)
	{
		auto headerBuf = imageBuf[0..Header.sizeof];
		if (!stdin.readExactly(headerBuf))
			return;

		auto pHeader = cast(Header*)headerBuf.ptr;
		imageBuf.length = pHeader.bfSize;
		auto dataBuf = imageBuf[Header.sizeof..$];
		enforce(stdin.readExactly(dataBuf), "Unexpected end of stream");

		if (pHeader.bcBitCount == 32)
		{
			// discard alpha
			auto imageAlpha = imageBuf.viewBMP!BGRX();
			imageAlpha.colorMap!(c => BGR(c.b, c.g, c.r)).copy(image);
		}
		else
			imageBuf.parseBMP!BGR(image);

	}
}

mixin main!(funopt!server);

bool readExactly(ref File f, ubyte[] buf)
{
	auto read = f.rawRead(buf);
	if (read.length==0) return false;
	enforce(read.length == buf.length, "Unexpected end of stream");
	return true;
}
*/
