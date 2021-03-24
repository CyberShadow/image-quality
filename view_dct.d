import std.algorithm.comparison;
import std.file;
import std.format;
import std.path;
import std.stdio;

import ae.sys.file;
import ae.utils.funopt;
import ae.utils.graphics.color;
import ae.utils.graphics.im_convert;
import ae.utils.graphics.image;
import ae.utils.main;
import ae.utils.meta;

import data;

void program(
	string inputFile,
)
{
	auto image = inputFile.read().parseViaIMConvert!BGR();
	auto imageDCT = Image!RGBf(image.w, image.h);

	foreach (by; 0 .. (image.h + 7) / 8)
		foreach (bx; 0 .. (image.w + 7) / 8)
			foreach (chan; 0 .. numChannels)
			{
				ubyte[basicBlockSize][basicBlockSize] block;
				float[basicBlockSize][basicBlockSize] result;

				foreach (cy; 0 .. basicBlockSize)
					foreach (cx; 0 .. basicBlockSize)
					{
						auto ix = bx * basicBlockSize + cx;
						auto iy = by * basicBlockSize + cy;
						if (ix < image.w && iy < image.h)
							block[cy][cx] = image[ix, iy].channel(chan);
					}
				dct(block, result);
				foreach (cy; 0 .. basicBlockSize)
					foreach (cx; 0 .. basicBlockSize)
					{
						auto ix = bx * basicBlockSize + cx;
						auto iy = by * basicBlockSize + cy;
						if (ix < image.w && iy < image.h)
						{
							auto f = result[cy][cx];
							imageDCT[ix, iy].channel(chan) = f;
							f = f.I!max(-1.0f).min(+1.0f); // -1 .. 1
							f /= 2; // -0.5 .. 0.5
							f += 0.5; // 0 .. 1
							image[ix, iy].channel(chan) = cast(ubyte)(255 * f);
						}
					}
			}

	image.colorMap!(c => RGB(c.r, c.g, c.b)).toPNG.toFile(inputFile.stripExtension ~ ".dct.png");

	foreach (chan; 0 .. numChannels)
	{
		auto f = File("%s.dct-%d.txt".format(inputFile.stripExtension, chan), "wb");

		foreach (y; 0 .. image.h)
		{
			foreach (x; 0 .. image.w)
				f.writef("%7.5f ", imageDCT[x, y].channel(chan));
			f.writeln();
		}
	}
}

mixin main!(funopt!program);
