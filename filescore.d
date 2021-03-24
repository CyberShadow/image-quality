import std.file;
import std.format;
import std.stdio;

import ae.utils.funopt;
import ae.utils.graphics.color;
import ae.utils.graphics.im_convert;
import ae.utils.main;

import server;

void fileScore(/*string modelFn, */string[] fns...)
{
	auto server = Server(/*modelFn, */ExecutionMode.cpuMultiThreaded, false);

	foreach (fn; fns)
		writeln(fn, ": ", server.analyzeImage(fn).score);
}

mixin main!(funopt!fileScore);
