import std.algorithm.comparison;
import std.algorithm.iteration;
import std.file;
import std.format;
import std.math;
import std.path;
import std.range;
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
	string baseName,
	uint targetIndex,
)
{
	int blockWidth = 8;
	int blockHeight = 8;

	auto indices = cast(uint[])read(baseName ~ "-indices.u32");
	auto labels = cast(bool[])read(baseName ~ "-labels.u8");
	auto pixels = cast(ubyte[])read(baseName ~ "-pixels-none.u8");
	auto metadata = cast(float[])read(baseName ~ "-metadata.f32");

	auto numMetadataVars = metadata.length / labels.length;

	Image!LA[2] images;
	foreach (which; [false, true])
	{
		auto sampleIndices = indices.length.iota.filter!(i => indices[i] == targetIndex && labels[i] == which).array;

		auto gridSize = cast(xy_t)ceil(sqrt(double(sampleIndices.length)));

		auto cellWidth = blockWidth + 2;
		auto cellHeight = blockHeight + 2;

		images[which].size(
			gridSize * cellWidth,
			gridSize * cellHeight,
		);

		foreach (i, sampleIndex; sampleIndices)
		{
			auto sx = 1 + (i % gridSize) * cellWidth;
			auto sy = 1 + (i / gridSize) * cellHeight;

			auto pixelIndex = sampleIndex * (blockWidth * blockHeight);

			foreach (py; 0 .. blockHeight)
				foreach (px; 0 .. blockWidth)
					images[which][sx + px, sy + py] = LA(pixels[pixelIndex++], 0xFF);
		}

		foreach (i; sampleIndices)
			writeln(metadata[i * numMetadataVars .. (i + 1) * numMetadataVars]);
		writeln("=========================");
	}
	foreach (ref image; images)
		image.size(image.w, max(images[0].h, images[1].h));
	auto filler = Image!LA(32, images[0].h);

	hjoin([
		images[0],
		filler,
		images[1]
	]).toPNG.toFile(format("%s-samples-%d.png", baseName, targetIndex));
}

mixin main!(funopt!program);
