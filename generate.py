#!/usr/bin/python3


from itertools import chain
from pathlib import Path
from ast import parse as ast_parse, literal_eval, unparse

from tokens import *
from specification import *


def roman(n:int)->str:
	"Make a Roman numeral."
	
	roman_numerals = {1000:'M', 900:'CM', 500:'D', 400:'CD', 100:'C', 90:'XC', 50:'L', 40:'XL', 10:'X', 9:'IX', 5:'V', 4:'IV', 1:'I'}
	roman_numeral = ''
	for value, numeral in sorted(roman_numerals.items(), reverse=True):
		while n >= value:
			roman_numeral += numeral
			n -= value
	return roman_numeral


def render_references(node, path, index, **kwargs):
	if path[-1] == 'Reference':
		if node.content[0] == "Table" or all(isinstance(_el, Digits) or _el == "." for _el in node.content):
			yield node.href
	else:
		for key, value in kwargs.items():
			if value is None:
				continue
			for r in value:
				if isinstance(r, str):
					if r.startswith('#'):
						yield r
				else:
					#print("key", key)
					try:
						try:
							for q in r:
								if isinstance(q, str):
									yield q
								else:
									yield from q
						except TypeError:
							yield from r
					except TypeError:
						pass


def render_cmd(index, content):
	"Make a string representing a single 'command' (i.e. sentence) from ECMA spec. Render all markup as ASCII."
	
	assert not isinstance(content, str|SpecialSymbol)
	result = []
	for token in content:
		if isinstance(token, str|SpecialSymbol):
			if token == "!":
				result.append("<raise_if_abrupt>")
			elif token == "?":
				result.append("<return_if_abrupt>")
			elif isinstance(token, SpecialSymbol):
				if token == SpecialSymbol.Sup:
					result.append("**")
				elif token == SpecialSymbol.Sub:
					pass
				elif token == SpecialSymbol.Bra:
					result.append("(")
				elif token == SpecialSymbol.Ket:
					result.append(")")
				else:
					raise NotImplementedError
			else:
				result.append(token)
		else:
			for subtoken in token:
				result.append(subtoken)
	
	for n in range(len(result)):
		if result[n] == "<function>" and n < len(result) - 2 and result[n + 2] == "of":
			result[n] = "<method>"
		elif result[n] == "<function>" and n < len(result) - 1 and result[n + 1] in ["`Contains`", "`ℝ`"]:
			result[n] = "<method>"
		elif result[n] == "<function>" and n < len(result) - 1 and result[n + 1] == "`modulo`":
			result[n] = ""
	yield None, index, " ".join(result)


def render_node(node, path, index, **kwargs):
	"Make a string representation of an object from ECMA grammar (i.e. algorithm, table, reference etc.)."
	
	if path[-1] == 'Cmd':
		yield from render_cmd(index, kwargs['content'])
		#yield path, index, " ".join(render_cmd(kwargs['content']))
		datum = kwargs['datum']
		if datum:
			yield from datum
	elif path[-1] in ['Terminal', 'Nonterminal']:
		if node.value in ["(", ")", "[", "]"]:
			yield path[-1] + "(chr(" + str(ord(node.value)) + "))"
		else:
			yield path[-1] + "(" + repr(node.value) + ")"
	elif path[-1] == 'Enum_':
		yield "<constant> " + repr(node.value)
	elif path[-1] in ['Variable', 'ObjectField']:
		yield "`" + node.value + "`"
	elif path[-1] == 'Constant':
		if node.value == "true" or node.value == "false" or node.value[0] == '"':
			yield node.value
		else:
			yield "`" + node.value.capitalize().replace('-', '_') + "`"
	elif path[-1] in ['Digits', 'HexDigits']:
		yield node.value
	elif path[-1] == 'Option':
		yield path[-1] + "(" + repr(node.type_) + ", " + repr(node.arg) + ")"
	elif path[-1] == 'Aoid':
		yield "<function>"
		yield "`" + node.aoid + "`"
	elif path[-1] == 'Reference':
		if all(_el == "." or isinstance(_el, (HexDigits, Digits)) for _el in node.content):
			yield "Section ["
			section = True
		else:
			section = False

		if kwargs['content'] is not None:
			for token in kwargs['content']:
				if isinstance(token, str):
					yield token
				else:
					yield from token
		
		if section:
			yield "]"
	
	elif path[-1] == 'Production':
		head = next(kwargs['head'])
		symbols = "[" + ", ".join(chain.from_iterable(kwargs['symbols'])) + "]"
		options = kwargs['options']
		if options is not None:
			options = "[" + ", ".join(("[" + ", ".join(chain.from_iterable(_options)) + "]") for _options in options) + "]"
		
		result = path[-1] + "(" + head + ", " + symbols + ((", " + options) if options is not None else "") + ")"
		
		if len(path) >= 3 and path[-2] == 'Grammar' and path[-3] == 'Clause':
			yield '<grammar>', index, result
		else:
			yield result
	elif path[-1] == 'Lookahead':
		productions = kwargs['productions']
		if productions is not None:
			productions = "[" + ", ".join(chain.from_iterable(productions)) + "]"
		yield path[-1] + "(" + repr(node.type_) + ((", " + productions) if productions is not None else "") + ")"
	elif path[-1] == 'Equation':
		# TODO
		yield '<equation>'
	elif path[-1] == 'Table':
		# TODO
		yield '<table>', (), ""
	elif path[-1] == 'Paragraph':
		for kind, idx, command in chain.from_iterable(kwargs['commands']):
			assert isinstance(command, str)
			yield '<paragraph>', idx, command
	elif path[-1] == 'Algorithm':
		for kind, idx, instruction in chain.from_iterable(kwargs['instructions']):
			assert isinstance(instruction, str)
			yield '<algorithm>', idx, instruction
	elif path[-1] == 'Grammar' and len(path) >= 2 and path[-2] == 'Clause':
		productions = kwargs['productions']
		if productions is not None:
			productions = chain.from_iterable(productions)
		for production in productions:
			assert isinstance(production, tuple) and len(production) == 3
			yield production
	elif path[-1] == 'Grammar':
		productions = kwargs['productions']
		if productions is not None:
			productions = ", ".join(chain.from_iterable(productions))
		else:
			productions = "<empty_grammar>"
		yield productions
	elif 'Cmd' in path or 'Paragraph' in path or 'Production' in path:
		raise ValueError(".".join(path))
			
	elif path[-1] == 'Note':
		pass
	else:
		raise NotImplementedError(".".join(path))


def render_scattered_sdo(node, path, index, **kwargs):
	"Iterate through all clauses and print out a 'scattered' syntax-directed operation, one that is spread across many clauses. There are 2 such functions: Evaluate and Early Errors."
	
	if path[-1] == 'Clause':
		prevkind = None
		sdo = False
		header = False
		for kind, idx, line in chain.from_iterable(kwargs['paragraphs']):
			if kind == '<paragraph>' and prevkind == None:
				yield line
				header = True
			
			if kind != '<paragraph>':
				header = False
			
			if kind == '<grammar>':
				if prevkind != '<grammar>':
					if prevkind != None:
						if sdo:
							#yield ""
							yield "Otherwise, in case this production does not match any of the Productions at the beginning, return `NotImplemented` at the end of the function."
						yield None
					yield "In case this production matches " + line
				else:
					yield " or this production matches " + line
				
				if prevkind in [None, '<paragraph>']:
					sdo = True
			elif kind == '<algorithm>' or (kind == '<paragraph>' and not header):
				if prevkind == '<grammar>':
					yield "  perform the following steps:"
				idx = idx[1:]
				yield " " * len(idx) + "(" + ".".join(str(_d + 1) for _d in idx) + ".) " + line
			#elif kind == '<paragraph>':
			#	yield line
			prevkind = kind
		
		for line in chain.from_iterable(kwargs['subclauses']):
			yield line
		
		if sdo:
			#yield ""
			yield "Otherwise, in case this production does not match any of the Productions at the beginning, return `NotImplemented` at the end of the function."
		yield None
	
	else:
		yield from render_node(node, path, index, **kwargs)


def specification_early_errors(specification):
	"Prepare the spec of Early Errors. Yields all subsections of Early Errors one by one."
	
	for clause in specification.find_clauses_with_title("Static Semantics: Early Errors"):
		result = []
		for line in clause.render(render_scattered_sdo):
			if line is not None:
				result.append(line)
			else:
				if result:
					yield "\n".join(result)
				result.clear()


def specification_evaluation(specification):
	"Prepare the spec of Evaluation. Yields all subsections of Evaluation one by one."
	
	for clause in specification.find_clauses_with_title("Runtime Semantics: Evaluation"):
		if not clause.paragraphs:
			continue
		if isinstance(clause.paragraphs[0], Paragraph) and clause.paragraphs[0].commands[0].content[:5] == ["The", "syntax", "-", "directed", "operation"]:
			continue
		result = []
		for line in clause.render(render_scattered_sdo):
			if line is not None:
				result.append(line)
			else:
				if result:
					yield "\n".join(result)
				result.clear()


def specification_abstract_operations(specification):
	"Prepare the spec of abstract operations. Yields all operations from the spec one by one."
	
	for clause in specification.find_clauses_with_title_ending_with(")"):
		if ("::" in clause.title) or ("." in clause.title):
			continue
		#if all(not clause.title.startswith(_name + " ") for _name in problematic_functions):
		#	continue
		#if all(not clause.title.startswith(_name + " ") for _name in ["BoundFunctionCreate"]):
		#	continue
		if not clause.paragraphs:
			continue
		
		found = False
		for n in range(len(clause.paragraphs)):
			if isinstance(clause.paragraphs[n], Paragraph):
				found = True
				break
			elif not isinstance(clause.paragraphs[n], Note):
				found = False
				break
		if not found:
			continue
		
		if isinstance(clause.paragraphs[n], Paragraph) and clause.paragraphs[n].commands and clause.paragraphs[n].commands[0].content[:3] != ["The", "abstract", "operation"]:
			continue
		
		result = []
		for line in clause.render(render_scattered_sdo):
			if line is None:
				if result:
					#for ref in frozenset(clause.render(render_references)):
					#	try:
					#		section = specification.find_definition(ref[1:])
					#	except KeyError:
					#		print("Href not found:", ref)
					#		continue
					#	else:
					#		result.append("")
					#		if isinstance(section, Table):
					#			result.append("# Table ")
					#		else:
					#			result.append("# Section [" + " . ".join(str(_x) for _x in section.chapter) + "]")
					#		for lv, ln in section._print():
					#			result.append(" " * lv + ln)
					#		result.append("")
						
					yield 'body', "\n".join(result)
				
				result.clear()
			elif line.startswith("The abstract operation "):
				assert not result
				yield 'head', line
			else:
				result.append(line)
		

def specification_syntax_directed_operations(specification):
	"Prepare the spec of syntax-directed operations. Yields all operations from the spec. First yields the operation header, then all its subsections one by one, then the header of the next operation and so on."
	
	for clause in chain(specification.find_clauses_with_title_starting_with("Static Semantics:"), specification.find_clauses_with_title_starting_with("Runtime Semantics:")):
		if clause.title.endswith(": SV") or clause.title.endswith(": MV") or clause.title.endswith(": TV") or clause.title.endswith(": TRV"):
			continue
		if not clause.paragraphs:
			continue
		
		found = False
		for n in range(len(clause.paragraphs)):
			if isinstance(clause.paragraphs[n], Paragraph):
				found = True
				break
			elif not isinstance(clause.paragraphs[n], Note):
				found = False
				break
		if not found:
			continue
		
		if isinstance(clause.paragraphs[n], Paragraph) and clause.paragraphs[n].commands and clause.paragraphs[n].commands[0].content[:5] != ["The", "syntax", "-", "directed", "operation"]:
			continue
		
		result = []
		for line in clause.render(render_scattered_sdo):
			if line is not None:
				result.append(line)
			else:
				if result:
					yield "\n".join(result)
				result.clear()


max_syntax_retries = 4


def generate_header(codegen, prompt, spec):
	"Ask AI to generate a header, that is the prototype of a function from its spec."
	
	personality = prompt[""] + prompt["Prototype"]
	retries = max_syntax_retries
	while retries:
		retries -= 1
		result = codegen.request(personality, spec, '{"__kind":').strip()
		print(result)
		try:
			tree = ast_parse(result)
			types = {}
			for key, value in zip(tree.body[0].value.keys, tree.body[0].value.values):
				key = literal_eval(key)
				if key.startswith('__'):
					value = literal_eval(value)
				else:
					value = unparse(value)
				types[key] = value
		except (SyntaxError, AttributeError, IndexError, ValueError):
			print("Syntax error!")
			spec += " \nDon't make syntax errors! Place quotes around strings correctly! (\"STRING\")"
			continue
		else:
			keys = frozenset(types.keys())
			forbidden_keys = frozenset({'list', 'object', 'input', 'global', 'match', 'len', 'exec', 'set', 'map', 'next', 'min', 'max', 'str', 'type', 'async'})
			if keys & forbidden_keys:
				spec += " \nAppend underscore to the following argument names: " + ", ".join(keys & forbidden_keys) + "."
				continue
			break
	else:
		raise RuntimeError("Model makes too many syntax errors!")
	
	func_kind = types['__kind']
	func_name = types['__name']
	func_args = []
	func_arg_types = []
	func_arg_optional = []
	optional = False
	func_return_type = ''
	for arg, type_ in types.items():
		if arg.startswith('__'):
			continue
		
		if arg == 'return':
			func_return_type = type_
		else:
			func_args.append(arg)
			if type_.endswith('None'):
				optional = True
			func_arg_optional.append(optional)
			func_arg_types.append(type_)
	
	return func_kind, func_name, func_args, func_arg_types, func_arg_optional, func_return_type


def generate_early_errors(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate Early Errors."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Early Errors"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			ast_parse(code)
		except SyntaxError as error:
			errors.append(error)
			spec += " \n\nDo not make syntax errors. Use correct indentation."
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def generate_evaluation(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate Evaluation."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Evaluation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			ast_parse(code)
		except SyntaxError as error:
			errors.append(error)
			spec += " \n\nDo not make syntax errors. Use correct indentation."
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def generate_condition(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate a single subsection of a syntax-directed operation."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Abstract Operation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			ast_parse(code)
		except SyntaxError as error:
			errors.append(error)
			spec += " \n\nDo not make syntax errors. Use correct indentation."
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def generate_algorithm(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate a single abstract operation."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Abstract Operation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			ast_parse(code)
		except SyntaxError as error:
			errors.append(error)
			print(code)
			spec += " \n\nDo not make syntax errors. Use correct indentation."
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def tee_files(*filenames, print_=True):
	def decor(old_fn):
		def new_fn(*args):
			fds = [Path(_filename).open('w') for _filename in filenames]
			if print_:
				fds.append(None)
			try:
				for text in old_fn(*args):
					for fd in fds:
						print(text, file=fd)
			finally:
				for fd in fds:
					if fd is not None:
						fd.close()
		return new_fn
	return decor


def compile_abstract_operations(specification, codegen, prompt):
	yield "#!/usr/bin/python3"
	yield ""
	yield ""
	yield "from definitions import *"
	yield ""
	yield ""
	for stype, spec in specification_abstract_operations(specification):
		if stype == 'head':
			this_spec = '\t"""' + spec.replace("\\", "\\\\").replace("\n", " ").strip() + '"""'
			func_kind, func_name, func_args, func_arg_types,  func_arg_optional, func_return_type = generate_header(codegen, prompt, spec)
			#assert func_kind == "abstract operation", func_kind
		elif stype == 'body':
			yield '"""'
			yield spec.replace("\\", "\\\\")
			yield '"""'
			if 'CompletionRecord' in func_return_type:
				yield "@ensure_completion"
			for n, line in enumerate(generate_algorithm(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type).split("\n")):
				if n == 1:
					yield this_spec
				yield line
			yield ""
			yield ""
		else:
			raise ValueError
	yield ""
	yield ""


def compile_early_errors(specification, codegen, prompt):
	yield "#!/usr/bin/python3"
	yield ""
	yield ""
	yield "__all__ = 'early_errors',"
	yield ""
	yield "from definitions import *"
	yield "from ao_library import *"
	yield ""
	yield ""
	final_function = []
	final_function.append("def early_errors(self, goal):")
	n = 0
	for spec in specification_early_errors(specification):
		yield '"""'
		yield spec.replace("\\", "\\\\")
		yield '"""'
		n += 1
		name = '_early_errors_' + roman(n).lower()
		yield generate_early_errors(codegen, prompt, spec, name, ('self', 'goal'), ('Self', 'Nonterminal'), (False, False), '')
		final_function.append(f"\tif {name}(self, goal) is not NotImplemented: return")
		yield ""
		yield ""
	yield from final_function
	yield "\traise NotImplementedError('early_errors')"
	yield ""
	yield ""


def compile_evaluation(specification, codegen, prompt):
	yield "#!/usr/bin/python3"
	yield ""
	yield ""
	yield "__all__ = 'Evaluation',"
	yield ""
	yield "from definitions import *"
	yield "from ao_library import *"
	yield ""
	yield ""
	final_function = []
	final_function.append("@ensure_completion")
	final_function.append("def Evaluation(self):")
	n = 0
	for spec in specification_evaluation(specification):
		yield '"""'
		yield spec.replace("\\", "\\\\")
		yield '"""'
		n += 1
		name = '_Evaluation_' + roman(n).lower()
		yield generate_evaluation(codegen, prompt, spec, name, ('self',), ('Self',), (False,), '')
		final_function.append(f"\tif (result := {name}(self)) is not NotImplemented: return result")
		yield ""
		yield ""
	yield from final_function
	yield "\traise NotImplementedError('Evaluation')"
	yield ""
	yield ""


def compile_syntax_directed_operations(specification, codegen, prompt):
	yield "#!/usr/bin/python3"
	yield ""
	yield ""
	yield "__all__ = 'SyntaxDirectedOperations',"
	yield ""
	yield "from definitions import *"
	yield "from ao_library import *"
	yield ""
	yield ""
	n = None
	final_class = []
	final_function = []
	for spec in specification_syntax_directed_operations(specification):
		if spec.startswith("The syntax - directed operation "):
			if final_function:
				final_function.append(f"\traise NotImplementedError('{func_name}')")
				final_function = []
			n = 0
			func_kind, func_name, func_args, func_arg_types, func_arg_optional, func_return_type = generate_header(codegen, prompt, spec)
			#assert func_kind == "syntax-directed operation", func_kind
			final_class.append(final_function)
			if 'CompletionRecord' in func_return_type:
				final_function.append("@ensure_completion")
			prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':'
			final_function.append(prototype)
			final_function.append('\t"""' + spec.replace("\\", "\\\\").replace("\n", " ").strip() + '"""')
		else:
			n += 1
			yield '"""'
			yield spec.replace("\\", "\\\\")
			yield '"""'
			name = '_' + func_name + '_' + roman(n).lower()
			yield generate_condition(codegen, prompt, spec, name, func_args, func_arg_types, func_arg_optional, func_return_type)
			final_function.append(f"\tif (result := {name}({', '.join(func_args)})) is not NotImplemented: return result")
			yield ""
			yield ""
	if final_function:
		final_function.append(f"\traise NotImplementedError('{func_name}')")
	yield "class SyntaxDirectedOperations:"
	for fun in final_class:
		for line in fun:
			yield "\t" + line
		yield "\t"
	yield ""
	yield ""



#if __debug__ and __name__ == '__main__':
#	from pickle import load
#	
#	with Path('specification.pickle').open('rb') as fd:
#		specification = load(fd)
#	
#	for stype, spec in specification_abstract_operations(specification):
#		if stype == 'head':
#			print("----")
#		print(spec)
#		print()
#
#	quit()


if __name__ == '__main__':
	from pickle import load
	from codegen import Codegen
	from lxml.etree import fromstring as xml_frombytes, tostring as xml_tobytes, tounicode as xml_tounicode
	from os import environ
	
	prompt_doc = xml_frombytes(Path('prompt.xml').read_bytes())
	prompt = {}
	for child in prompt_doc:
		context = child.attrib.get('context', "")
		text = [child.text]
		for subchild in child:
			text.append(subchild.tail)
		prompt[context] = "\n".join(text)
	del prompt_doc
	
	
	with Path('specification.pickle').open('rb') as fd:
		specification = load(fd)
	
	print("URL:  ", environ['LLM_API_URL'])
	print("model:", environ['LLM_MODEL'])
	
	codegen = Codegen(url=environ['LLM_API_URL'], api_key=environ['LLM_API_KEY'])
	codegen.configure(**eval(environ['LLM_CONFIG_EXTRA']))
	#print(list(codegen.list_models()))
	codegen.configure(model=environ['LLM_MODEL'], rate_limit=4, min_delay=3, rate_limit_down=0.05, rate_limit_up=0.75)
	
	dest_dir = Path('gencode')
	dest_dir.mkdir(exist_ok=True)

	problematic_functions = ["Set", "PutValue", "CreateArrayIterator", "DoWait", "NewPromiseReactionJob", "MinFromTime", "TimeWithinDay", "Day", "GetValueFromBuffer", "SetValueInBuffer", "GetRawBytesFromSharedBlock"]
	
	verbose = True
	#tee_files(dest_dir / 'ao_library.py', print_=verbose)(compile_abstract_operations)(specification, codegen, prompt)
	#tee_files(dest_dir / 'early_errors.py', print_=verbose)(compile_early_errors)(specification, codegen, prompt)
	#tee_files(dest_dir / 'evaluation.py', print_=verbose)(compile_evaluation)(specification, codegen, prompt)
	tee_files(dest_dir / 'sdo_library.py', print_=verbose)(compile_syntax_directed_operations)(specification, codegen, prompt)

