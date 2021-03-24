import std.algorithm.iteration;
import std.algorithm.searching;
import std.conv : to;
import std.exception;
import std.file;
import std.format;
import std.math;
import std.parallelism;
import std.path;
import std.process;
import std.random;
import std.range;
import std.stdio;

import ae.sys.datamm;
import ae.sys.file;
import ae.utils.graphics.color;
import ae.utils.graphics.im_convert;
import ae.utils.graphics.image;
import ae.utils.json;

import metadata;
import data : countColors, FileFormat;
import tests : numTests, TestInfo;

void main()
{
	stderr.writeln("Cleaning up...");
	foreach (de; dirEntries("tests/", SpanMode.shallow))
		de.remove();

	auto images = "images".dirEntries("*.{png,jp*,bmp,gif}", SpanMode.shallow).map!(de => de.name).array;

	static string toBmp(string fn)
	{
		auto bmp = fn.setExtension(".bmp");
		enforce(spawnProcess([
			"convert",
			"-background", "white",
			"-alpha", "remove",
			"-type", "TrueColor",
			fn ~ "[0]", bmp,
		]).wait() == 0, "convert failed");
		return bmp;
	}

	static void checkBmp(string fn)
	{
		auto data = mapFile(fn, MmMode.read);
		auto bmp = viewBMP!BGR(data.contents);
		enforce(bmp.w > 8 && bmp.h > 8, "Image width/height is too small");
		//enforce(bmp.w * bmp.h > 8*8 * 128, "Image area is too small");
	}

	// // Convert a uniform distribution between [0,1] into something
	// // that resembles a bell curve centered around 0.5.
	// // `pow` is the strength of the curvature (i.e. how much more the
	// // 0.5 vicinity is preferred over the edges) and must be a
	// // positive odd integer.
	// double centerDist(double x, int pow = 3)
	// {
	// 	return ((2 * x - 1) ^^ pow + 1) / 2;
	// }

	// // Returns an unbounded normal distribution centered around 0
	// double randomNormal()
	// {
	// 	return sqrt(log(uniform01!double) * -2) * cos(2 * PI * uniform01!double);
	// 	// return normalDistribution(uniform01!double);
	// }

	// Returns an normal-ish distribution centered around 0.5 and bounded to (0,1)
	double randomNormal01()
	{
		// auto x = randomNormal();
		// x = x / (1 + abs(x));
		// // x = x / 2 + 0.5;
		// return x;

		// auto x = uniform01!double;
		// return (acos(((1 - x) - x)) / PI);

		auto y = uniform01!double;
		return (sin(acos(1 - y * 2))*((y <0.5) * 2 - 1) + ((y >= 0.5) * 2)) / 2;
	}

	int randomScale() // returns percent
	{
		// auto r = randomNormal();
		// return cast(int)(10 + (55*2) * randomNormal());

		// return cast(int)round(randomNormal01 * 100);
		// return cast(int)round(10 + randomNormal01 * (55 * 2));

		enum q = 1.0;
		enum x = q / 0.65;
		enum d = 2.5 * q;
		auto y = x - (d/2) + d * randomNormal01();
		return cast(int)(100 * q / y);
	}

	int randomQuality() // returns percent
	{
		auto x = randomNormal01;
		x = x * 2 - 1;
		x = abs(x);
		x = 1 - x;
		return cast(int)round(100 * x);
	}

	auto ok = new bool[numTests];
	uint pass = 0;
	do
	{
		writeln("=======================================================================");

		foreach (n; numTests.iota.parallel(1))
		{
			if (ok[n])
				continue;

			try
			{
				rndGen.seed(pass * numTests + n);

				string img = images[pass * numTests + n];

				stderr.writeln(img);
				auto origExt = img.extension;
				auto origFn = format("tests/%04d-orig%s", n, origExt);
				hardLink(img, origFn);
				scope(failure) origFn.remove();

				TestInfo.Image imageInfo(string fileName)
				{
					TestInfo.Image image;
					image.fileName = fileName;
					auto i = fileName
						.read
						.parseViaIMConvert!BGR;
					image.width = i.w.to!uint;
					image.height = i.h.to!uint;
					image.fileSize = getSize(fileName);
					image.format = getFileFormat(fileName);
					image.quality = getQuality(fileName);
					image.colorCount = countColors(i);
					return image;
				}

				TestInfo info;
				info.images[0] = imageInfo(origFn);

				auto origBmp = toBmp(origFn);
				scope(failure) origBmp.remove();
				checkBmp(origBmp);

				string[] cmdLine;
				bool lossy;

				static string[] interpolate()
				{
					return [
						"-interpolate", 
						[
							"average", "average4", "average9", "average16",
							/*"background",*/ "bilinear", "blend", "catrom",
							"integer", "mesh", "nearest-neighbor", "spline"
						][uniform(0, $)]
					];
				}

				static string[] filter()
				{
					return [
						"-filter", 
						[
							// simple
							"Point",     "Hermite",     "Cubic",
							"Box",       "Gaussian",    "Catrom",
							"Triangle",  "Quadratic",   "Mitchell",
							"CubicSpline",

							// windowed
							"Lanczos",     "Hamming",     "Parzen",
							"Blackman",    "Kaiser",      "Welsh",
							"Hanning",     "Bartlett",    "Bohman",
						][uniform(0, $)]
					];
				}

				foreach (count; 0 .. uniform(0, 2+1))
				{
					auto resize = [
						{
							return ["-scale", "%d%%".format(randomScale())];
						},
						{
							return interpolate ~ [
								"-define", "sample:offset=%d".format([0, 50, 100][uniform(0, $)]),
								"-sample", "%d%%".format(randomScale()),
							];
						},
						{
							return filter ~ interpolate ~ [
								"-resize", "%d%%".format(randomScale()),
							];
						},
					//	{
					//		return ["-magnify"];
					//	},
					//	{
					//		return filter ~ interpolate ~ [
					//			"-liquid-rescale", "%d%%".format(uniform(50, 150)),
					//		];
					//	},
						{
							return filter ~ interpolate ~ [
								"+distort", "SRT", "%f,0".format(randomScale() / 100.0),
							];
						},
					][uniform(0, $)]();

					[
						{
							resize = resize;
						},
						{
							resize = ["-gamma", ".45455"] ~ resize ~ ["-gamma", "2.2"];
						},
						{
							resize = ["-colorspace", "RGB"] ~ resize ~ ["-colorspace", "sRGB"];
						},
					][uniform(0, $)]();

					cmdLine ~= resize;
					lossy = true;
				}

				[
					{
						cmdLine = cmdLine;
					},
					{
						cmdLine = cmdLine ~ ["-shave", "%d%%".format(uniform(1, 10))];
						lossy = true;
					},
				][uniform(0, $)]();

				FileFormat targetFormat = getFileFormat(images[uniform(0, numTests)]);

				if (uniform(0, 10) == 0 || targetFormat == FileFormat.GIF)
				{
					cmdLine = cmdLine ~
						[
							"-dither", ["None", "FloydSteinberg", "Riemersma"][uniform(0, $)],
							"-colors", "%d".format(uniform(64, 256))
						];

					if (info.images[0].colorCount > 256)
						lossy = true;
				}

				string editExt;
				final switch (targetFormat)
				{
					case FileFormat.PNG:
						editExt = ".png";
						break;
					case FileFormat.JPEG:
						cmdLine ~= ["-quality", "%d".format(randomQuality)];
						lossy = true;
						editExt = ".jpg";
						break;
					case FileFormat.GIF:
						editExt = ".gif";
						break;
				}

				enforce(lossy, "Chosen parameters do not incur lossy transformations"); // Retry

				auto editFn = format("tests/%04d-edit%s", n, editExt);
				cmdLine = ["convert", origBmp] ~ cmdLine ~ [editFn];
				writeln(cmdLine);
				enforce(spawnProcess(cmdLine).wait() == 0, "convert failed");
				scope(failure) editFn.remove();
				auto editBmp = toBmp(editFn);
				scope(failure) editBmp.remove();
				checkBmp(editBmp);

				info.images[1] = imageInfo(editFn);
				info.imCmdLine = cmdLine;

				auto infoFn = format("tests/%04d-info.json", n);
				File(infoFn, "wb").writeln(info.toJson);

				ok[n] = true;
			}
			catch (Exception e)
				writeln(e);
		}

		pass++;
	} while (!all(ok));
}
