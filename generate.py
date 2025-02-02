#!/usr/bin/python3



from random import choice, randint


class CodegenMock:
	def __init__(self, **kwargs):
		pass
	
	def init(self):
		pass
	
	def __random_str(self):
		result = ""
		while ":" not in result:
			result = "".join(choice(":::  _abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _n in range(randint(40, 95)))
		return result
	
	def request(self, system, user, prefix=None):
		if prefix is None:
			prefix = ""
		return prefix + "\n\t".join(self.__random_str() for _n in range(randint(5, 15)))
	
	def finish(self):
		pass



from itertools import chain
from pathlib import Path

from tokens import *
from specification import *


def roman(n:int)->str:
	roman_numerals = {1000:'M', 900:'CM', 500:'D', 400:'CD', 100:'C', 90:'XC', 50:'L', 40:'XL', 10:'X', 9:'IX', 5:'V', 4:'IV', 1:'I'}
	roman_numeral = ''
	for value, numeral in sorted(roman_numerals.items(), reverse=True):
		while n >= value:
			roman_numeral += numeral
			n -= value
	return roman_numeral


def render_cmd(index, content):
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
		yield path[-1] + "[" + repr(node.value) + "]"
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
		if kwargs['content'] is not None:
			for token in kwargs['content']:
				if isinstance(token, str):
					yield token
				else:
					yield from token
		#yield "/" + node.href + "/"
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
							yield ""
							yield "Otherwise (`self` does not match the Production above) return `NotImplemented`."
						yield None
					yield "If this production matches " + line
				else:
					yield " or this production matches " + line
				
				if prevkind in [None, '<paragraph>']:
					sdo = True
			elif kind == '<algorithm>' or (kind == '<paragraph>' and not header):
				if prevkind == '<grammar>':
					yield "then:"
				idx = idx[1:]
				yield " " * len(idx) + "(" + ".".join(str(_d + 1) for _d in idx) + ".) " + line
			#elif kind == '<paragraph>':
			#	yield line
			prevkind = kind
		
		for line in chain.from_iterable(kwargs['subclauses']):
			yield line
		
		if sdo:
			yield ""
			yield "Otherwise (`self` does not match the Production above) return `NotImplemented`."
		yield None
	
	else:
		yield from render_node(node, path, index, **kwargs)


def specification_early_errors(specification):
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
	for clause in specification.find_clauses_with_title_ending_with(")"):
		if ("::" in clause.title) or ("." in clause.title) or ("EnumerateObjectProperties " in clause.title) or ("CreateIntrinsics " in clause.title):
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
		
		if isinstance(clause.paragraphs[n], Paragraph) and clause.paragraphs[n].commands and clause.paragraphs[n].commands[0].content[:3] != ["The", "abstract", "operation"]:
			continue
		
		result = []
		for line in clause.render(render_scattered_sdo):
			if line is None:
				if result:
					yield "\n".join(result)
				result.clear()
			elif line.startswith("The abstract operation "):
				assert not result
				yield line
			else:
				result.append(line)


def specification_syntax_directed_operations(specification):
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


def generate_header(codegen, prompt, spec):
	personality = prompt[""] + prompt["Prototype"]
	result = codegen.request(personality, spec).strip()
	print(result)
	#raise NotImplementedError
	func_kind, func_name, *argtypes = eval(result) # FIXME: safe eval
	func_args = []
	func_arg_types = []
	func_arg_optional = []
	func_return_type = ''
	for argtype in argtypes:
		n = argtype.index(":")
		arg = argtype[:n].strip()
		type_ = argtype[n+1:].strip()
		
		if arg == 'return':
			func_return_type = type_
		else:
			func_args.append(arg)
			if type_.endswith('/optional'):
				func_arg_optional.append(True)
				type_ = type_[:-9].strip()
			else:
				func_arg_optional.append(False)
			func_arg_types.append(type_)
	
	return func_kind, func_name, func_args, func_arg_types, func_arg_optional, func_return_type


def generate_early_errors(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Early Errors"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	return codegen.request(personality, spec, prototype)


def generate_evaluation(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Evaluation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	return codegen.request(personality, spec, prototype)


def generate_condition(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	return codegen.request(personality, spec, prototype)


def generate_algorithm(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	personality = prompt[""] + prompt["Algorithm"] + prompt["Abstract Operation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	return codegen.request(personality, spec, prototype)


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
	for spec in specification_abstract_operations(specification):
		if spec.startswith("The abstract operation "):
			this_spec = '\t"""' + spec.replace("\\", "\\\\").replace("\n", " ").strip() + '"""'
			func_kind, func_name, func_args, func_arg_types,  func_arg_optional, func_return_type = generate_header(codegen, prompt, spec)
			#assert func_kind == "abstract operation", func_kind
		else:
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
		final_function.append(f"\tif (result := {name}(self, goal)) is not NotImplemented: return")
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


if __name__ == '__main__':
	from pickle import load
	from codegen import Codegen
	from lxml.etree import fromstring as xml_frombytes, tostring as xml_tobytes, tounicode as xml_tounicode
	from os import environ
	
	prompt_doc = xml_frombytes(Path('prompts/mistral.xml').read_bytes())
	model = prompt_doc.attrib['model']
	temperature = float(prompt_doc.attrib['temperature'])
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
	
	codegen = Codegen('https://api.groq.com/openai/v1', api_key=environ['GROQ_API_KEY'])
	codegen.configure(model='gemma2-9b-it', prepend_prefix=True, rate_limit=3, min_delay=2, rate_limit_down=0.05, rate_limit_up=0.5)
	verbose = True
	tee_files('gencode/ao_library.py', print_=verbose)(compile_abstract_operations)(specification, codegen, prompt)
	tee_files('gencode/early_errors.py', print_=verbose)(compile_early_errors)(specification, codegen, prompt)
	tee_files('gencode/evaluation.py', print_=verbose)(compile_evaluation)(specification, codegen, prompt)
	tee_files('gencode/sdo_library.py', print_=verbose)(compile_syntax_directed_operations)(specification, codegen, prompt)

