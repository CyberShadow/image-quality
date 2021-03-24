module data;

import std.algorithm.comparison : min;
import std.algorithm.iteration;
import std.algorithm.mutation;
import std.algorithm.setops;
import std.algorithm.sorting;
import std.bitmanip;
import std.conv;
import std.exception;
import std.experimental.allocator.mmap_allocator;
import std.format;
import std.math;
import std.random;
import std.range;
import std.traits;
import std.typecons : Tuple;

import ae.utils.appender;
import ae.utils.graphics.color;
import ae.utils.graphics.image;
import ae.utils.graphics.view;
import ae.utils.range;

enum numChannels = 3;
ref typeof(COLOR.r) channel(COLOR)(in ref COLOR c, uint chan) { return (cast(typeof(COLOR.r)*)&c)[chan]; }

enum basicBlockSize = 8;

void dct(in ref ubyte[basicBlockSize][basicBlockSize] block, out float[basicBlockSize][basicBlockSize] result)
{
	byte[basicBlockSize][basicBlockSize] input;
	foreach (v; 0 .. basicBlockSize)
		foreach (u; 0 .. basicBlockSize)
			input[v][u] = cast(byte)(block[v][u] - 128);

	static real alpha(size_t u) { enum SQRT2_INV = 1 / SQRT2; return u ? 1 : SQRT2_INV; }

	static immutable float[basicBlockSize][basicBlockSize] cosTable = {
		float[basicBlockSize][basicBlockSize] result;
		foreach (x; 0 .. basicBlockSize)
			foreach (u; 0 .. basicBlockSize)
				result[x][u] = cos(((x * 2 + 1) * u * PI) / 16);
		return result;
	}();

	foreach (v; 0 .. basicBlockSize)
		foreach (u; 0 .. basicBlockSize)
			result[v][u] = 0;

	foreach (y; 0 .. basicBlockSize)
		foreach (x; 0 .. basicBlockSize)
		{
			auto p = input[y][x];
			auto cx = cosTable[x].ptr;
			auto cy = cosTable[y].ptr;
			foreach (v; 0 .. basicBlockSize)
			{
				auto p1 = p * cy[v];
				foreach (u; 0 .. basicBlockSize)
					result[v][u] += p1 * cx[u];
			}
		}

	foreach (v; 0 .. basicBlockSize)
		foreach (u; 0 .. basicBlockSize)
			result[v][u] *= 0.25 * alpha(u) * alpha(v);
}

uint countColors(V)(in ref V img)
if (isView!V)
{
	static BitArray seenColor;
	seenColor.length = 256 * 256 * 256;
	seenColor[] = false;
	foreach (y; 0 .. img.h)
		foreach (e; img.scanline(y))
			foreach (c; e)
				seenColor[
					c.r * 256 * 256 +
					c.g * 256 +
					c.b
				] = true;

	return cast(uint)seenColor.count;
}

enum Preprocessing
{
	none,
	derivative,
	dct,
}

enum Format
{
	none,
	nul,
	u8,
	u32,
	f32,
}

enum Order
{
	sequential,
	random,
	entropy,
}

alias BGRf = Color!(float, "b", "g", "r");

struct DataBuffer
{
	alias Allocator = MmapAllocator;
	FastAppender!(ubyte, Allocator) buf;
	Format format;

	void put(T)(T val)
	{
		static if (is(typeof(val) == ubyte))
		{
			final switch (format)
			{
				case Format.none:
					assert(false);
				case Format.nul:
					return;
				case Format.u8:
					return buf.put((&val)[0..1]);
				case Format.u32:
					return put(uint(val));
				case Format.f32:
					return put(float(val) / 255f);
			}
		}
		else
		static if (is(typeof(val) == uint))
		{
			final switch (format)
			{
				case Format.none:
					assert(false);
				case Format.nul:
					return;
				case Format.u8:
				case Format.f32:
					throw new Exception("Can't write " ~ T.stringof ~ " to " ~ text(format) ~ " file");
				case Format.u32:
					return buf.put(cast(ubyte[])(&val)[0..1]);
			}
		}
		else
		static if (is(typeof(val) == float))
		{
			assert(val == val, "NaN found!");
			final switch (format)
			{
				case Format.none:
					assert(false);
				case Format.nul:
					return;
				case Format.u8:
				case Format.u32:
					throw new Exception("Can't write " ~ T.stringof ~ " to " ~ text(format) ~ " file");
				case Format.f32:
					return buf.put(cast(ubyte[])(&val)[0..1]);
			}
		}
		else
			static assert(false, "Unknown type: " ~ T.stringof);
	}
}

enum FileFormat
{
	PNG,
	JPEG,
	GIF,
}

struct ImageProcessor
{
	uint blockWidth;
	uint blockHeight;
	Preprocessing pre;
	uint imageSize;
	Order sampleOrder;
	uint imageSamples;
	double imageSamplesFraction;
	bool filterSamples;

	uint loadImage(V)(
		in ref V img,
		bool label, ulong fileSize, uint index,
		FileFormat fileFormat, float quality,
		ref DataBuffer pixels,
		ref DataBuffer labels,
		ref DataBuffer metadata,
		ref DataBuffer indices,
		DataBuffer* entropies,
	) const
	if (isView!V)
	{
		if (fileFormat != FileFormat.JPEG)
			quality = 1;

		auto view = img.crop(
			0, 0,
			min(imageSize ? imageSize : int.max, img.w),
			min(imageSize ? imageSize : int.max, img.h),
		);

		auto colorCount = countColors(img);
		uint numImageSamples = 0;

		static uint entropy(S)(ref S sample)
		{
			auto pixels = sample.getOrigPixels();
			return zip(pixels, pixels.dropOne).map!(pair => abs(pair[0] - pair[1])).sum;
		}

		void addSample()
		{
			metadata.put(float(1f / img.w     ));
			metadata.put(float(1f / fileSize  ));
			metadata.put(float(1f / colorCount));
			metadata.put(float(     quality   ));
			foreach (format; EnumMembers!FileFormat)
				metadata.put(float(format == fileFormat));
			indices.put(index);
			labels.put(ubyte(label));
			numImageSamples++;
		}

		// Process selected samples
		void doSamples(R)(R samples)
		{
			auto buffer = &pixels;
			foreach (sample; samples)
			{
				addSample();
				foreach (pixel; sample.getPixels)
					buffer.put(pixel);

				{
					static BitArray seenColor;
					seenColor.length = 256;
					seenColor[] = false;
					foreach (pixel; sample.getOrigPixels)
						seenColor[pixel] = true;
					auto colorCount = seenColor.count;
					metadata.put(float(1f / colorCount));
				}

				{
					auto sampleEntropy = entropy(sample);
					metadata.put(float(1f / (1 + sampleEntropy)));
					if (entropies)
						entropies.put(sampleEntropy);
				}
			}
		}

		// Limit blocks
		void doLimit(R)(R samples)
		{
			alias doNext = doSamples;

			if (imageSamples)
				doNext(samples.take(imageSamples));
			else
			if (imageSamplesFraction)
			{
				static ElementType!R[] buf;
				auto arr = samples.bufArray(buf);
				doNext(arr.take(cast(size_t)ceil(arr.length * imageSamplesFraction)));
			}
			else
				doNext(samples);
		}

		// Filter blocks
		void doFilter(R)(R samples)
		{
			alias doNext = doLimit;

			if (filterSamples)
				doNext(samples.filter!(
						(sample)
						{
							auto pixels = sample.getOrigPixels();
							auto p0 = pixels.front;
							foreach (p; pixels.dropOne)
								if (p != p0)
									return true;
							return false;
						}));
			else
				doNext(samples);
		}

		// Select blocks
		void doOrder(R)(R samples)
		{
			alias doNext = doFilter;
			alias S = Unqual!(ElementType!R);

			static S[] buf;

			final switch (sampleOrder)
			{
				case Order.sequential:
					return doNext(samples);
				case Order.random:
				{
					auto arr = samples.bufArray(buf);
					auto mut = cast(Unqual!(typeof(arr[0]))[])arr; // strip const
					mut.randomShuffle();
					return doNext(mut);
				}
				case Order.entropy:
				{
					auto arr = samples.bufArray(buf);
					static uint[] entropiesBuf;
					auto entropies = arr.map!entropy.bufArray(entropiesBuf);
					static size_t[] orderBuf;
					auto order = arr.length.iota.bufArray(orderBuf);
					order.sort!((a, b) => entropies[a] > entropies[b]);
					return doNext(order.map!(index => arr[index]));
				}
			}
		}

		// Split into blocks
		void doBlocks(O, P)(O orig, P preprocessed)
		{
			alias doNext = doOrder;

			// Canonicalize image size across all preprocessing transforms
			// - Subtract 1 for Preprocessing.derivative
			// - Round down to multiple of basicBlockSize for Preprocessing.dct
			auto w = (cast(uint)orig.w - 1) / basicBlockSize * basicBlockSize;
			auto h = (cast(uint)orig.h - 1) / basicBlockSize * basicBlockSize;

			auto getViewPixels(V)(V v, uint bx, uint by, uint chan) @nogc
			{
				static struct CoordPred
				{
					/**/ V v; uint blockWidth; uint blockHeight; uint bx; uint by; uint chan;
					this(V v, uint blockWidth, uint blockHeight, uint bx, uint by, uint chan)
					{ this.v = v; this.blockWidth = blockWidth; this.blockHeight = blockHeight; this.bx = bx; this.by = by; this.chan = chan; }

					auto opCall(Tuple!(uint, uint) ic)
					{
						return v[
							bx * blockWidth  + ic[0],
							by * blockHeight + ic[1],
						].channel(chan);
					}
				}
				auto coordPred = CoordPred(v, blockWidth, blockHeight, bx, by, chan);
				auto bw = blockWidth  ? blockWidth  : w;
				auto bh = blockHeight ? blockHeight : h;
				return
					cartesianProduct(
						bw.iota,
						bh.iota,
					)
					.takeExactly(bw * bh)
					.pmap(coordPred);
			}

			struct PixelGetter
			{
				uint bx, by, chan;

				auto getPixels()     { return getViewPixels(preprocessed, bx, by, chan); }
				auto getOrigPixels() { return getViewPixels(orig        , bx, by, chan); }
			}

			auto numBlocksX = blockWidth  ? w / blockWidth  : 1;
			auto numBlocksY = blockHeight ? h / blockHeight : 1;
			return doNext(
				cartesianProduct(
					numBlocksX.iota,
					numBlocksY.iota,
				)
				.takeExactly(numBlocksX * numBlocksY)
				.map!(c => PixelGetter(c[0], c[1], (c[0] * 4294967291U + c[1]) % numChannels))
			);
		}

		// Apply preprocessing
		void doPre(V)(V v)
		{
			alias doNext = doBlocks;

			final switch (pre)
			{
				case Preprocessing.none:
					doNext(v, v);
					break;
				case Preprocessing.derivative:
				{
					static Image!BGRf nDer;
					alias n = nDer;
					n.size(v.w, v.h);

					foreach (chan; 0 .. numChannels)
						foreach (y; 0 .. v.h)
							foreach (x; 0 .. v.w - 1)
							{
								auto c0 = v[x    , y];
								auto c1 = v[x + 1, y];
								n[x, y].channel(chan) = (float(c1.channel(chan)) - float(c0.channel(chan))) / 255;
							}
					doNext(v, n);
					break;
				}
				case Preprocessing.dct:
				{
					static Image!BGRf nDCT;
					alias n = nDCT;
					n.size(v.w, v.h);

					foreach (chan; 0 .. numChannels)
						foreach (y; 0 .. v.h / basicBlockSize)
							foreach (x; 0 .. v.w / basicBlockSize)
							{
								ubyte[basicBlockSize][basicBlockSize] block;
								foreach (j; 0 .. basicBlockSize)
									foreach (i; 0 .. basicBlockSize)
										block[j][i] = v[x * basicBlockSize + i, y * basicBlockSize + j].channel(chan);
								float[basicBlockSize][basicBlockSize] coeffs = void;
								dct(block, coeffs);
								foreach (j; 0 .. basicBlockSize)
									foreach (i; 0 .. basicBlockSize)
										n[x * basicBlockSize + i, y * basicBlockSize + j].channel(chan) = abs(coeffs[j][i]);
							}
					doNext(v, n);
					break;
				}
			}
		}

		// Scan in each direction, if applicable
		void doDirections()
		{
			alias doNext = doPre;

			doNext(view);
			if (pre == Preprocessing.derivative)
				doNext(view.flipXY);
		}

		doDirections();

		enforce(numImageSamples > 0,
			"Test %s %s image has no samples"
			.format(index, ["orig", "edit"][label]));

		return numImageSamples;
	}
}

unittest
{
	if (false)
	{
		ImageProcessor ip;
		Image!RGB image;
		DataBuffer buffer;
		ip.loadImage(
			image,
			false, 0, 0,
			FileFormat.init, 0,
			buffer,
			buffer,
			buffer,
			buffer,
			null,
		);
	}
}

A bufArray(R, A)(R r, ref A buf)
if (is(A == Unqual!(ElementType!R)[]))
{
	static if (is(R == A))
		return r;
	else
	static if (hasLength!R)
	{
		if (buf.length < r.length)
			buf.length = r.length;
		auto copied = copy(r, buf);
		assert(r.length + copied.length == buf.length);
		return buf[0 .. r.length];
	}
	else
	{
		size_t i = 0;
		foreach (ref e; r)
		{
			if (i == buf.length)
				buf.length = buf.length ? buf.length * 2 : 16;
			buf[i++] = e;
		}
		return buf[0 .. i];
	}
}
