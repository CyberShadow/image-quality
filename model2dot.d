import std.algorithm.iteration;
import std.format;
import std.stdio;
import dyaml;

void main(string[] args)
{
	auto fn = args[1];
	Node root = Loader.fromFile(fn).load();

	auto f = File(fn ~ ".dot", "wb");
	f.writeln("digraph {");
	foreach (Node layer; root["config"]["layers"])
	{
		auto name = layer["name"].as!string;
		auto desc = "%s(%-(%s, %))".format(
			layer["class_name"].as!string,
			["units", "filters", "kernel_size", "rate", "activation"]
			.filter!(p => p in layer["config"])
			.map!(p => layer["config"][p].as!string));
		f.writefln("\t%s [label=%(%s%)]", name, [desc]);
		foreach (Node i; layer["inbound_nodes"])
			foreach (Node c; i)
				f.writefln("\t%s -> %s;", c[0].as!string, name);
	}
	f.writeln("}");
	f.flush();
}
