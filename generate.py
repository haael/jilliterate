#!/usr/bin/python3


from itertools import chain
from pathlib import Path
from ast import parse as ast_parse, literal_eval, unparse
from collections import defaultdict, Counter
from hashlib import sha256
from inspect import getfullargspec
from pickle import load, dump
from lxml.etree import fromstring as xml_frombytes, tostring as xml_tobytes, tounicode as xml_tounicode
from enum import Enum
from types import UnionType

from tokens import *
from specification import *


CSI = "\x1B["


pickle_dir = Path('pickle')

forbidden_identifiers = frozenset({'len', 'min', 'next'})


class Category(Enum):
	informal = 0
	function = 1
	type_ = 2
	normative = 3
	
	@classmethod
	def typed_eval(cls, s):
		s = unparse(s)
		if isinstance(s, str) and s.startswith(cls.__name__ + '.'):
			return cls[s[len(cls.__name__) + 1:]]
		raise ValueError(f"Could not convert to {cls.__name__}: {repr(s)}")


class Kind(Enum):
	unknown = 0
	
	abstract_operation = 1
	syntax_directed_operation = 2
	method = 3
	abstract_method = 4
	constructor = 5
	operator = 6
	
	plain = 11
	struct = 12
	enum = 13
	union = 14
	
	property_ = 21
	slot = 22
	relation = 23
	
	@classmethod
	def typed_eval(cls, s):
		s = unparse(s)
		if isinstance(s, str) and s.startswith(cls.__name__ + '.'):
			return cls[s[len(cls.__name__) + 1:]]
		raise ValueError(f"Could not convert to {cls.__name__}: {repr(s)}")


class TypeName:
	__type_names = frozenset()
	
	def __class_getitem__(cls, values):
		try:
			orig_cls = cls.orig_cls
		except AttributeError:
			orig_cls = cls
		
		if values is Ellipsis:
			return type(cls.__name__, (cls,), {f'_{cls.__name__}__type_names':Ellipsis, 'orig_cls':orig_cls})
		else:
			return type(cls.__name__, (cls,), {f'_{cls.__name__}__type_names':cls.__type_names | values, 'orig_cls':orig_cls})
	
	def __new__(cls, name):
		try:
			orig_cls = cls.orig_cls
		except AttributeError:
			orig_cls = cls
		
		r = object.__new__(orig_cls)
		r.__init__(name)
		return r
	
	def __getnewargs__(self):
		return self.__name,
	
	def __init__(self, name):
		self.__name = name
	
	def __repr__(self):
		return self.__name
	
	def __eq__(self, other):
		try:
			return self.__name == other.__name
		except AttributeError:
			return NotImplemented
	
	@classmethod
	def typed_eval(cls, s):
		s = unparse(s)
		if (cls.__type_names is Ellipsis) or (s in cls.__type_names):
			return cls(s)
		else:
			raise ValueError(f"Invalid type name: {s}. Allowed: {cls.__type_names}")


def typed_eval(type_, tree):
	if type_ is None:
		if tree is None:
			return None
		else:
			raise TypeError
	elif isinstance(type_, UnionType):
		errors = []
		for t in type_.__args__:
			try:
				return typed_eval(t, tree)
			except Exception as error:
				errors.append(error)
		else:
			raise ExceptionGroup("Typed eval failed on union type.", errors)
	else:
		try:
			r = literal_eval(tree)
		except ValueError:
			r = type_.typed_eval(tree)
		else:
			if not isinstance(r, type_):
				raise TypeError(f"{repr(r)} of type {type(r)} is not an instance of {type_}")
		
		return r


def render_cmd(index, content):
	"Make a string representing a single 'command' (i.e. sentence) from ECMA spec. Render all markup as ASCII."
	
	#print("rc", content)
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
				assert isinstance(token, str)
				result.append(token)
		else:
			#raise NotImplementedError(str(type(token)))
			for subtoken in token:
				assert isinstance(subtoken, str)
				result.append(subtoken)
	
	for n in range(len(result)):
		if result[n] == "<function>" and n < len(result) - 2 and result[n + 2] == "of":
			result[n] = "<method>"
		elif result[n] == "<function>" and n < len(result) - 1 and result[n + 1] in ["`Contains`", "`ℝ`"]:
			result[n] = "<method>"
		elif result[n] == "<function>" and n < len(result) - 1 and result[n + 1] == "`modulo`":
			result[n] = ""
		elif result[n] == "<function>" and n < len(result) - 2 and result[n + 2] != "(":
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
		yield "<enum> `" + str(node.value) + "`"
		#yield "`Enum_." + node.value.replace('-', '_') + "`"
	elif path[-1] in ['Variable', 'ObjectField']:
		yield "`" + node.value + "`"
	elif path[-1] == 'Constant':
		if node.value == "true" or node.value == "false" or node.value[0] == '"':
			yield node.value
		else:
			yield "`" + node.value.replace('-', '_') + "`"
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
	elif path[-1] == 'Definition':
		# TODO
		yield '<definition>'
	elif path[-1] == 'Table':
		# TODO
		yield '<table>', (), "==== Table"
		n = 0
		for row in kwargs['head']:
			k = []
			for cell in row:
				k.append(" ".join([str(_s) for _, _, _s in cell]))
			yield '<table>', (n,), "\t" + " | ".join(k) + ""
			n += 1
		yield '<table>', (), "\t----"
		for row in kwargs['body']:
			k = []
			for cell in row:
				k.append(" ".join([str(_s) for _, _, _s in cell]))
			yield '<table>', (n,), "\t" + " | ".join(k) + ""
			n += 1
		yield '<table>', (), "===="
		#yield '<table>', (), ""
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


def pickle_cache(*args_to_hash):
	def decorate(old_fun):
		fun_args = getfullargspec(old_fun).args
		arg_indices = [fun_args.index(_ath) for _ath in args_to_hash]
		
		def new_fun(*args, **kwargs):
			pickle_cache_recreate = kwargs['_recreate']
			pickle_cache_contains = kwargs['_contains']
			
			pickle_file = pickle_dir / (old_fun.__name__ + '_' + sha256(repr(([args[_n] for _n in arg_indices], )).encode('utf-8')).hexdigest()[2:34] + '.pickle')
			
			recreate = False
			if not recreate: recreate = (pickle_cache_recreate is Ellipsis)
			if not recreate: recreate = (pickle_cache_recreate is not None) and (old_fun.__name__ in pickle_cache_recreate)
			if not recreate and pickle_cache_contains:
				for n in arg_indices:
					try:
						if any((_token in args[n]) for _token in pickle_cache_contains):
							recreate = True
							break
					except TypeError:
						pass
			
			try:
				if recreate:
					pickle_file.unlink()
				else:
					with pickle_file.open('rb') as fd:
						return load(fd)
			except IOError:
				pass
			
			result = old_fun(*args, **kwargs)
			
			with pickle_file.open('wb') as fd:
				dump(result, fd)
			
			return result
		
		return new_fun
	
	return decorate


@pickle_cache('spec_prompt')
def generate_dicts(codegen, retries, system_prompt, spec_prompt, error_prompts, key_types, extra_keys_exclude, extra_keys_type, _recreate=None, _contains=None):
	"Ask an AI to create a dict (one or more)."
	
	if _recreate is not None and _recreate is not Ellipsis and 'generate_dicts' in _recreate:
		_recreate = ...
	elif _recreate is not Ellipsis and _contains is not None and any((_token in spec_prompt) for _token in _contains):
		_recreate = ...
	
	prefix_prompt = '{' + repr(next(iter(key_types.keys()))) + ":"
	
	result = []
	errors = []
	while retries >= 0:
		retries -= 1
		
		response = codegen.request(system_prompt, spec_prompt, prefix_prompt)
		#print("dict response:", repr(response))
		
		try:
			tree = ast_parse(response)
			
			if tree.__class__.__name__ != 'Module':
				raise ValueError("Expected 'Module' result.")
			if len(tree.body) == 0:
				raise ValueError("Expected nonempty result.")
			if any(_expr.__class__.__name__ != 'Expr' for _expr in tree.body):
				raise ValueError("Expected 'Expr' as each element of the module body.")
			if any(_expr.value.__class__.__name__ != 'Dict' for _expr in tree.body):
				raise ValueError("Expected 'Dict' as value of each expression in the module.")
			
			for expr in tree.body:
				dict_ = {}
				
				for key, value in zip(expr.value.keys, expr.value.values):
					key = literal_eval(key)
					
					try:
						keytype = key_types[key]
					except KeyError:
						if extra_keys_exclude is not None and extra_keys_type is not None:
							if key in extra_keys_exclude:
								raise KeyError(f"Extra key not allowed: `{key}`.")
							try:
								dict_[key] = typed_eval(extra_keys_type, value)
							except Exception as error:
								raise ValueError(f"Value could not be parsed: {unparse(value)}") from error
						else:
							raise
					else:
						try:
							dict_[key] = typed_eval(keytype, value)
						except Exception as error:
							raise ValueError(f"Value could not be parsed: {unparse(value)}") from error
						pass
				
				for key in frozenset(key_types.keys()) - frozenset(dict_.keys()):
					if not isinstance(None, key_types[key]):
						raise KeyError(f"Missing key `{key}` in returned dict.")
				
				result.append(dict_)
		
		except (SyntaxError, ValueError, KeyError, TypeError) as error:
			print(CSI + "33m" + repr(response) + CSI + "0m")
			print(CSI + "31;40m" + type(error).__name__ + ":" + CSI + "0m" + CSI + "31m" + " " + str(error) + CSI + "0m")
			try:
				spec_prompt += error_prompts[len(errors)]
			except IndexError:
				pass
			errors.append(error)
		
		else:
			return result
	else:
		raise ExceptionGroup("Too many errors while making a dict.", errors)


@pickle_cache('spec_prompt', 'func_name')
def generate_function(codegen, retries, system_prompt, spec_prompt, error_prompts, func_name, arg_types, return_type, variables_exclude, _recreate=None, _contains=None):
	"Ask an AI to create a function."
	
	if _recreate is not None and _recreate is not Ellipsis and 'generate_function' in _recreate:
		_recreate = ...
	elif _recreate is not Ellipsis and _contains is not None and any((_token in spec_prompt or _token == func_name) for _token in _contains):
		_recreate = ...
	
	prefix_prompt = 'def ' + func_name + '(' + ', '.join(((_arg + ':' + repr(_type)) if (_type is not None) else _arg) for (_arg, _type) in arg_types.items()) + ')' + ((' -> ' + repr(return_type)) if (return_type is not None) else '') + ':\n\t'
	
	errors = []
	while retries >= 0:
		retries -= 1
		
		response = codegen.request(system_prompt, spec_prompt, prefix_prompt)
		
		try:
			tree = ast_parse(response)
			
			if tree.__class__.__name__ != 'Module':
				raise ValueError("Expected 'Module' result.")
			if len(tree.body) != 1:
				raise ValueError("Expected exactly one item in result.")
			if tree.body[0].__class__.__name__ != 'FunctionDef':
				raise ValueError("Expected 'FunctionDef' as the sole element of the module body.")
			
			func = tree.body[0]
			if func.name != func_name:
				raise ValueError("The returned function name 'tree.body[0].name' does not match the requested one '{func_name}'.")
			
			# TODO
		
		except (SyntaxError, ValueError, KeyError, TypeError) as error:
			print(CSI + "33m" + repr(response) + CSI + "0m")
			print(CSI + "31;40m" + type(error).__name__ + ":" + CSI + "0m" + CSI + "31m" + " " + str(error) + CSI + "0m")
			try:
				spec_prompt += error_prompts[len(errors)]
			except IndexError:
				pass
			errors.append(error)
		
		else:
			return func
	else:
		raise ExceptionGroup("Too many errors while making a dict.", errors)


@pickle_cache('spec')
def generate_spec(codegen, retries, prompt, spec, _recreate=None, _contains=None):
	"Ask AI to guess what the paragraph is about."
	
	if _recreate is not None and _recreate is not Ellipsis and 'generate_spec' in _recreate:
		#_recreate = ...
		pass
	elif _recreate is not Ellipsis and _contains is not None and any((_token in spec) for _token in _contains):
		_recreate = ...
	
	personality = prompt[""] + "\n\n" + prompt["Guess content"]
	arg_types = {'category':Category, 'kind':Kind | None, 'name':str | None, 'superclass':str | None, 'subtypes':list | None, 'values':list | None, 'specification_type':bool | None, 'implementation_defined':bool | None, 'class':str | None}
	error_prompts = []
	for n in range(retries):
		try:
			error_prompts.append(prompt[f"Guess content / Syntax Error {n + 1}"])
		except KeyError:
			error_prompts.append("")
	
	errors = []
	while retries >= 0:
		responses = generate_dicts(codegen, retries, personality, spec, error_prompts, arg_types, forbidden_identifiers, None, _recreate=_recreate, _contains=_contains)
		#print("responses:", repr(responses))
		_recreate = ...
		retries -= 1
		
		result = []
		try:
			for nn, response in enumerate(responses):
				#print("response", nn, response)
				category = response['category']
				
				try:
					if "." in response['name']:
						raise ValueError("Name must not contain dots.")
				except KeyError:
					pass
				
				if category == Category.informal:
					result.append([category])
				
				elif category == Category.function:
					kind = response.get('kind', Kind.unknown)
					if kind not in frozenset({Kind.unknown, Kind.abstract_operation, Kind.syntax_directed_operation, Kind.method, Kind.abstract_method, Kind.constructor, Kind.operator}):
						raise ValueError(f"{kind} is not a valid value for Category.function.")
					
					if kind == Kind.unknown:
						result.append([category, kind])
					elif kind in frozenset({Kind.abstract_operation, Kind.syntax_directed_operation, Kind.operator}):
						if response['class'] is not None:
							raise ValueError(f"'class' must be None.")
						result.append([category, kind, response['name'], response['implementation_defined'], None])
					elif kind in frozenset({Kind.method, Kind.abstract_method, Kind.constructor}):
						if response['class'] is None:
							raise ValueError(f"'class' is required for methods.")
						result.append([category, kind, response['name'], response['implementation_defined'], response['class']])
					else:
						raise ValueError(f"Unsupported kind: {kind}.")
				
				elif category == Category.type_:
					kind = response.get('kind', Kind.unknown)
					if kind not in frozenset({Kind.unknown, Kind.plain, Kind.struct, Kind.enum, Kind.union}):
						raise ValueError(f"{kind} is not a valid value for Category.type_.")
					
					if kind == Kind.unknown:
						result.append([category, kind])
					elif kind == Kind.plain:
						if response['subtypes'] is not None:
							raise ValueError(f"'subtypes' must be None.")
						if response['values'] is not None:
							raise ValueError(f"'values' must be None.")
						#if response['superclass'] is not None:
						#	raise ValueError(f"'superclass' must be None.")
						result.append([category, kind, response['name'], response['superclass'], response['subtypes'], response['specification_type']])
					elif kind == Kind.union:
						if response['subtypes'] is None:
							raise ValueError(f"'subtypes' required for union type.")
						if response['values'] is not None:
							raise ValueError(f"'values' must be None.")
						if response['superclass'] is not None:
							raise ValueError(f"'superclass' must be None.")
						result.append([category, kind, response['name'], None, response['subtypes'], response['specification_type']])
					elif kind == Kind.enum:
						if response['subtypes'] is not None:
							raise ValueError(f"'subtypes' must be None.")
						if response['values'] is None:
							raise ValueError(f"'values' required for enum type.")
						if response['superclass'] is not None:
							raise ValueError(f"'superclass' must be None.")
						result.append([category, kind, response['name'], None, response['values'], response['specification_type']])
					elif kind == Kind.struct:
						if response['values'] is not None:
							raise ValueError(f"'values' must be None.")
						if response['subtypes'] is not None:
							raise ValueError(f"'subtypes' must be None.")
						if (response['superclass'] is None) and (response['name'] not in ["Object", "Record"]):
							raise ValueError(f"'superclass' required for struct type.")
						if (response['superclass'] is not None) and (response['name'] in ["Object", "Record"]):
							raise ValueError(f"'superclass' must be None for Object and Record.")
						result.append([category, kind, response['name'], response['superclass'], response['values'], response['specification_type']])
					else:
						raise ValueError
				
				elif category == Category.normative:
					kind = response.get('kind', Kind.unknown)
					if kind not in frozenset({Kind.unknown, Kind.slot, Kind.property_, Kind.relation}):
						raise ValueError(f"{kind} is not a valid value for Category.normative.")
					
					if kind == Kind.unknown:
						result.append([category, kind])
					elif kind in frozenset({Kind.relation}):
						result.append([category, kind, response['name'], None])
					else:
						if response['class'] is None:
							raise ValueError(f"'class' is required for properties and slots.")
						result.append([category, kind, response['name'], response['class']])
				
				else:
					raise ValueError
		
		except (SyntaxError, ValueError, KeyError) as error:
			print(CSI + "33m" + repr(responses) + CSI + "0m")
			print(CSI + "31;40m" + type(error).__name__ + ":" + CSI + "0m" + CSI + "31m" + " " + str(error) + CSI + "0m")
			try:
				spec += error_prompts[len(errors)]
			except IndexError:
				pass
			errors.append(error)
		
		else:
			#print("result length", len(result))
			return result
	
	else:
		raise ExceptionGroup("Could not make a function prototype.", errors)


@pickle_cache('spec')
def generate_prototype(codegen, retries, prompt, spec, forbidden_identifiers, allowed_type_names, _recreate=None, _contains=None):
	"Ask AI to generate a header, that is the prototype of a function from its spec."
	
	if _recreate is not None and _recreate is not Ellipsis and 'generate_prototype' in _recreate:
		_recreate = ...
	elif _recreate is not Ellipsis and _contains is not None and any((_token in spec) for _token in _contains):
		_recreate = ...
	
	AllowedTypeName = TypeName[allowed_type_names]
	personality = prompt[""] + "\n\n" + prompt["Prototype"]
	arg_types = {'__kind':Kind, '__name':str, '__return':AllowedTypeName}
	error_prompts = [prompt[f"Prototype / Syntax Error {_n + 1}"] for _n in range(retries)]
	
	errors = []
	while retries >= 0:
		types = generate_dicts(codegen, retries, personality, spec, error_prompts, arg_types, forbidden_identifiers, AllowedTypeName, _recreate=_recreate, _contains=_contains)
		retries -= 1
		
		try:
			if len(types) != 1:
				raise ValueError
			types = types[0]
			
			func_kind = types['__kind']
			del types['__kind']
			
			func_name = types['__name']
			del types['__name']
			if func_name in forbidden_identifiers:
				raise ValueError
			
			try:
				return_type = types['__return']
				del types['__return']
			except KeyError:
				return_type = ''
		
		except (SyntaxError, ValueError) as error:
			print(CSI + "31;40m" + type(error).__name__ + ":" + CSI + "0m" + CSI + "31m" + " " + str(error) + CSI + "0m")
			try:
				spec += error_prompts[len(errors)]
			except IndexError:
				pass
			errors.append(error)
		
		else:
			return func_kind, func_name, types, return_type
	
	else:
		raise ExceptionGroup("Could not make a function prototype.", errors)


@pickle_cache('spec_prompt', 'docstring', 'func_name')
def generate_abstract_operation(codegen, retries, prompt, spec_prompt, docstring, func_name, arg_types, return_type, variables_exclude, _recreate=None, _contains=None):
	if _recreate is not None and _recreate is not Ellipsis and 'generate_abstract_operation' in _recreate:
		_recreate = ...
	elif _recreate is not Ellipsis and _contains is not None and any((_token in spec_prompt or _token == func_name) for _token in _contains):
		_recreate = ...
	
	system_prompt = prompt[""] + "\n\n" + prompt["Algorithm"] + "\n\n" + prompt["Abstract Operation"] + "\n\n" + prompt.get("Function / " + func_name, "")
	error_prompts = [prompt[f"Algorithm / Syntax Error {_n + 1}"] for _n in range(retries)]
	result = generate_function(codegen, retries, system_prompt, spec_prompt, error_prompts, func_name, arg_types, return_type, variables_exclude, _recreate=_recreate, _contains=_contains)
	result.body.insert(0, ast_parse(repr(docstring)).body[0])
	return result


def load_prompt():
	prompt_doc = xml_frombytes(Path('prompt.xml').read_bytes())
	prompt = {}
	for child in prompt_doc:
		if child.tag == 'prompt':
			context = child.attrib.get('context', "")
			text = [child.text]
			for subchild in child:
				text.append(subchild.tail)
			prompt[context] = "\n".join(text)
			
			for subchild in prompt_doc:
				if subchild.tag != 'line': continue
				subcontext = subchild.attrib.get('context', "")
				text = [subchild.text]
				for subsubchild in subchild:
					text.append(subsubchild.tail)
				prompt[context + " / " + subcontext] = "\n".join(text)
		elif child.tag == 'function':
			text = [child.text]
			for subchild in child:
				text.append(subchild.tail)
			prompt["Function / " + child.attrib['name']] = "\n".join(text)
	
	return prompt


if __name__ == '__main__':
	from codegen import Codegen
	from os import environ
	from sys import argv
	
	assert isinstance(Kind.unknown, Kind)
	assert eval("Kind.unknown") == Kind.unknown
	assert Kind.typed_eval(ast_parse("Kind.unknown")) == Kind.unknown
	
	pickle_dir.mkdir(exist_ok=True)
	if '--purge' in argv:
		for f in pickle_dir.iterdir():
			if f.suffix == '.pickle' and f.name != 'specification.pickle':
				f.unlink()
		quit()
	
	with (pickle_dir / 'specification.pickle').open('rb') as fd:
		specification = load(fd)
	
	prompt = load_prompt()
	
	print("URL:  ", environ['LLM_API_URL'])
	print("model:", environ['LLM_MODEL'])
	
	codegen = Codegen(url=environ['LLM_API_URL'], api_key=environ['LLM_API_KEY'])
	codegen.configure(**eval(environ['LLM_CONFIG_EXTRA']))
	codegen.configure(model=environ['LLM_MODEL'], rate_limit=4, min_delay=3, rate_limit_down=0.05, rate_limit_up=0.75)
	
	contains = set()
	recreate = set()
	if '--dicts' in argv: recreate.add('generate_dicts')
	if '--functions' in argv: recreate.add('generate_function')
	if '--specs' in argv: recreate.add('generate_spec')
	if '--prototypes' in argv: recreate.add('generate_prototype')
	if '--abstract-operations' in argv: recreate.add('generate_abstract_operation')
	if '--everything' in argv: recreate = ...
	for arg in argv:
		if arg.startswith('--recreate='):
			token = arg[11:]
			contains.add(token)
	
	#quit()
	
	if __debug__ and ('--sanity' in argv):
		#print(recreate)
		personality = "You are a code generator. Output syntactically correct Python code. Use tab for indent. Do not explain the code."
		
		r = generate_dicts(codegen, 0, personality, "Make a dict with keys: a=1, b=2, c=3.", [], {'a':int, 'b':int, 'c':int}, None, None, _recreate=recreate, _contains=contains)
		#print(r)
		assert [{'a':1, 'b':2, 'c':3}] == r
		
		f = generate_function(codegen, 0, personality, "Make a function with 2 arguments: x and y returning their sum.", [], 'add_floats', {'x':TypeName('float'), 'y':TypeName('float')}, TypeName('float'), forbidden_identifiers, _recreate=recreate, _contains=contains)
		res = {}
		exec(unparse(f), globals(), res) # FIXME: unsafe
		add_floats = res['add_floats']
		assert add_floats(3.0, 7.0) == 10.0
		
		AllowedTypeName = TypeName[{'int', 'str', 'Frobnicator'}]
		
		prototype_prompt = """
 Analyze the following spec line and output information about the specified fuction. Make a dict with keys: '__kind', '__name', '__return' and one key for each argument.
 '__kind' value should the `Kind.abstract_operation` (not a string, but expression).
 '__name' value should be the name of the function, as provided in the spec (a string).
 '__return' should be the return type (a valid Python expression).
 Also there should be one key for each argument, in the order as they appear in the spec, with the argument type as expression.
"""
		r = generate_dicts(codegen, 0, personality + prototype_prompt, "A function tabfoo takes an argument `a` (an integer) and `boo` (a Frobnicator) and returns a string.", [], {'__kind':Kind, '__name':str, '__return':AllowedTypeName}, forbidden_identifiers, AllowedTypeName, _recreate=recreate, _contains=contains)
		#print(r)
		assert r == [{'__kind': Kind.abstract_operation, '__name': 'tabfoo', '__return': AllowedTypeName('str'), 'a': AllowedTypeName('int'), 'boo': AllowedTypeName('Frobnicator')}]
		
		spec_prototype = "An abstract operation append_spaces takes an argument `s` (a str) and `n` (an int) and returns a str."
		spec_body = "The return value is the string `s` with `n` spaces appended at the end."
		
		kind, name, argtypes, rettype = generate_prototype(codegen, 0, prompt, spec_prototype, forbidden_identifiers, ..., _recreate=recreate, _contains=contains)
		assert kind == Kind.abstract_operation
		assert name == 'append_spaces'
		assert argtypes == {'s': TypeName("String"), 'n': TypeName("Number['int']")}
		assert rettype == TypeName('String')
		
		argtypes['s'] = TypeName('str')
		argtypes['n'] = TypeName('int')
		rettype = TypeName('str')
		rp = generate_abstract_operation(codegen, 0, prompt, spec_body, spec_prototype, name, argtypes, rettype, forbidden_identifiers, _recreate=recreate, _contains=contains)
		res = {}
		exec(unparse(rp), globals(), res) # FIXME: unsafe
		append_spaces = res['append_spaces']
		assert append_spaces("boom", 3) == "boom   "
		
		spec1 = "The String Type: The String type is the set of all ordered sequences of zero or more 16-bit unsigned integer values (“elements”) up to a maximum length of 2**53 - 1 elements. The String type is generally used to represent textual data in a running ECMAScript program, in which case each element in the String is treated as a UTF-16 code unit value."
		results = generate_spec(codegen, 0, prompt, spec1, _recreate=recreate, _contains=contains)
		assert results == [[Category.type_, Kind.plain, 'String', None, None, False]]
		
		spec2 = "The Boolean Constructor: The Boolean constructor: is %Boolean%."
		results = generate_spec(codegen, 0, prompt, spec2, _recreate=recreate, _contains=contains)
		assert results == [[Category.function, Kind.constructor, 'Boolean']]
		
		spec3 = "Numeric Types: ECMAScript has two built-in numeric types: Number and BigInt."
		results = generate_spec(codegen, 0, prompt, spec3, _recreate=recreate, _contains=contains)
		assert results == [[Category.type_, Kind.union, 'Numeric', None, ['Number', 'BigInt'], False]]
		
		spec4 = "The List and Record Specification Types: The List type is used to explain the evaluation of argument lists (see 13.3.8) in new expressions, in function calls, and in other algorithms where a simple ordered list of values is needed. Values of the List type are simply ordered sequences of list elements containing the individual values."
		results = generate_spec(codegen, 0, prompt, spec4, _recreate=recreate, _contains=contains)
		assert results == [[Category.type_, Kind.plain, 'List', None, None, True], [Category.type_, Kind.struct, 'Record', None, None, True]]
	
	
	def extended_title(specification, chapter):
		if not chapter:
			return []
		else:
			clause = specification.find_clause_with_chapter(chapter)
			title = clause.title
			return extended_title(specification, chapter[:-1]) + [title]
	
	type_hierarchy = defaultdict(set)
	unions = defaultdict(set)
	enums = defaultdict(set)
	classes = defaultdict(set)
	plains = defaultdict(set)
	#functions = defaultdict(set)
	attributes = defaultdict(lambda: defaultdict(set))
	
	do = False
	last_chapter = None
	for id_ in specification.subclause_ids():
		clause = specification.find_clause(id_)
		if clause.chapter:
			last_chapter = clause.chapter
			do = (6 <= clause.chapter[0])
		if not do: continue
		
		if len(clause.paragraphs) == 0:
			continue
		elif not isinstance(clause.paragraphs[0], Paragraph):
			continue
		elif len(clause.paragraphs[0].commands):
			lines = ["\t" + _l.strip() for (_, _, _l) in clause.paragraphs[0].render(render_node)]
		else:
			continue
		
		if not lines:
			continue
		
		etitle = extended_title(specification, last_chapter)
		if clause.title and etitle[-1] != clause.title:
			etitle.append(clause.title)
		
		spec = []
		spec.append("[ " + " / ".join(etitle[:-1]) + " ]")
		spec.append("")
		spec.append(etitle[-1] + ":")
		spec.append("")
		spec.extend(lines)
		
		text = "\n".join(spec)
		
		#if text.startswith("See") or text.startswith("This function") or text.startswith("This method"):
		#	etitle = extended_title(specification, last_chapter)
		#	spec = etitle[-2] + " / " + ((clause.title + ": ") if clause.title else "") + text
		#else:
		#	spec = ((clause.title + ": ") if clause.title else "") + text
		
		print('.'.join(str(_d) for _d in clause.chapter))
		print(CSI + "34;40m" + text + CSI + "0m")
		for nn, result in enumerate(generate_spec(codegen, 3, prompt, text, _recreate=recreate, _contains=contains)):
			print("generated spec:", '.'.join(str(_d) for _d in last_chapter), nn, CSI + "36m" + repr(result) + CSI + "0m")
			category = result[0]
			if category == Category.informal:
				continue
			kind = result[1]
			if kind == Kind.unknown:
				continue
			
			if category == Category.function:
				name = result[2]
				implementation_defined = result[3]
				class_name = result[4]
				if class_name is not None:
					attributes[class_name][kind].add(name)
				else:
					...
			
			elif category == Category.type_:
				name = result[2]
				superclass = result[3]
				specification_type = result[5]
				if specification_type:
					type_hierarchy['SpecificationType'].add(name)
					unions['SpecificationType'].add(name)
				
				if kind == Kind.union:
					subtypes = result[4]
					unions[name].update(subtypes)
					type_hierarchy[name].update(subtypes)
					if superclass:
						type_hierarchy[superclass].add(name)
				elif kind == Kind.enum:
					values = result[4]
					print(name, values)
					if values is not None:
						enums[name].update(values)
					if superclass:
						type_hierarchy[superclass].add(name)
				elif kind == Kind.struct:
					name = result[2]
					superclass = result[3]
					if superclass:
						type_hierarchy[superclass].add(name)
					classes[name].add(clause.id_)
				elif kind == Kind.plain:
					name = result[2]
					superclass = result[3]
					if superclass:
						type_hierarchy[superclass].add(name)
					plains[name].add(clause.id_)
				else:
					raise ValueError
			
			elif category == Category.normative:
				name = result[2]
				class_name = result[3]
				if class_name is not None:
					attributes[class_name][kind].add(name)
			
			else:
				pass
		print()
	
	# FIXME: change prompt instead
	type_hierarchy['Object'].add('ExoticObject')
	classes['ExoticObject']
	type_hierarchy['LanguageValue'].add('Numeric')
	#type_hierarchy['SpecificationType'].remove('ArrayIterator')
	
	# ECMA specification reuses same identifiers for different types, i.e. "Set" may mean set object, set specification type and the abstract operation "Set". Fix those ambiguities by appending "Object" to object types.
	c = Counter()
	for key, values in type_hierarchy.items():
		if key in values:
			values.remove(key)
		c.update(values)
	duplicates = frozenset(_key for _key, _freq in c.items() if _freq >= 2)
	
	duplicate_branches = []
	for duplicate in duplicates:
		duplicate_branches.append([duplicate])
	
	modified = True
	while modified:
		modified = False
		current_duplicate_branches = duplicate_branches
		duplicate_branches = []
		for branch in current_duplicate_branches:
			found = False
			head = branch[0]
			for key, values in type_hierarchy.items():
				if head in values:
					duplicate_branches.append([key] + branch)
					modified = True
					found = True
			if not found:
				duplicate_branches.append(branch)
	
	#for branch in sorted(duplicate_branches, key=lambda _x: _x[-1]):
	#	print(branch)
	
	for duplicate in duplicates:
		nonobject_parents = {}
		object_parents = {}
		
		for branch in duplicate_branches:
			if branch[-1] == duplicate:
				if 'Object' in branch:
					object_parents[branch[-2]] = len(branch)
				else:
					nonobject_parents[branch[-2]] = len(branch)
		
		#print(duplicate, object_parents, nonobject_parents)
		
		if (object_parents and nonobject_parents):
			for parent in list(object_parents.keys()):
				type_hierarchy[parent].remove(duplicate)
				type_hierarchy[parent].add(duplicate + 'Object')
				del object_parents[parent]
				classes[duplicate + 'Object'] = classes[duplicate]
				del classes[duplicate]
		
		longest = max(chain(object_parents.values(), nonobject_parents.values()))
		for parent, length in chain(object_parents.items(), nonobject_parents.items()):
			if length != longest:
				type_hierarchy[parent].remove(duplicate)
	
	#print(duplicates)
	
	middle_leaves = frozenset(chain.from_iterable(type_hierarchy.values()))
	middle_top = frozenset(type_hierarchy.keys())
	
	toplevel = middle_top - middle_leaves
	
	assert toplevel == frozenset({'LanguageValue', 'SpecificationType'}), str(list(toplevel))
	
	c = Counter()
	for key, values in type_hierarchy.items():
		if key in values:
			values.remove(key)
		c.update(values)
	duplicates = frozenset(_key for _key, _freq in c.items() if _freq >= 2)
	
	assert not duplicates, str(list(duplicates))
	
	#print(middle_leaves & middle_top)
	#print('SpecificationType' in middle_leaves)
	
	
	def print_tree(key, level=0):
		if key in classes:
			print(" " * level + key, "(struct)")
		elif key in unions:
			print(" " * level + key, "(union)")
		elif key in enums:
			print(" " * level + key, "(enum)")
		elif key in plains:
			print(" " * level + key, "(plain)")
		else:
			#print(" " * level + key, "(unknown)")
			raise ValueError("Uknown type: " + key)
		
		for value in sorted(type_hierarchy[key]):
			print_tree(value, level + 1)
	
	print("Type hierarchy:")
	for key in reversed(sorted(toplevel)):
		print_tree(key)
	print()
	#quit()
	
	for classname in sorted(attributes.keys()):
		#if '.' in classname:
		#	#c = classname.split('.')
		#	#if c[-1] == '__proto__': continue
		#	#if c[0] == 'well_known': del c[0]
		#	#if c[-1] == 'prototype': del c[-1]
		#	#classname = '.'.join(c)
		#	print(classname)
		#	#for values in methods[classname]:
		#	#	print("\t", *values)
		#	print()
		#
		#elif classname not in type_hierarchy:
		#	#raise ValueError(classname)
		#	pass
		#	print("missing", classname)
		if True:
			print(f"class {classname}:")
			for value in attributes[classname][Kind.slot]:
				print(f"\t{value}:object")
			for value in attributes[classname][Kind.property_]:
				print(f"\t{value}:object")
			for value in attributes[classname][Kind.constructor]:
				print("\t@classmethod")
				print(f"\tdef {value}(cls):")
				print("\t\tpass")
				print("\t")
			for value in attributes[classname][Kind.abstract_method]:
				print(f"\tdef {value}(self):")
				print("\t\tpass")
				print("\t")
			for value in attributes[classname][Kind.method]:
				print(f"\tdef {value}(self):")
				print("\t\tpass")
				print("\t")
			print()
	
	def generate_type(name, parent):
		if name in unions:
			pass
		elif name in enums:
			assert parent is None
			print()
			print(f"class {name}(Enum):")
			for value in enums[name]:
				print(f"\t{value} = auto()")
			print()
		elif name in classes:
			print()
			
			if parent is None:
				print(f"class {name}:")
			else:
				print(f"class {name}({parent}):")

			for value in attributes[classname][Kind.slot]:
				print(f"\t{value}:object")
			for value in attributes[classname][Kind.property_]:
				print(f"\t{value}:object")
			for value in attributes[classname][Kind.constructor]:
				print("\t@classmethod")
				print(f"\tdef {value}(cls):")
				print("\t\tpass")
				print("\t")
			for value in attributes[classname][Kind.abstract_method]:
				print(f"\tdef {value}(self):")
				print("\t\tpass")
				print("\t")
			for value in attributes[classname][Kind.method]:
				print(f"\tdef {value}(self):")
				print("\t\tpass")
				print("\t")
			
			#print("\t\"\"\"")
			#for id_ in classes[name]:
			#	clause = specification.find_clause(id_)
			#	for paragraph in clause.paragraphs:
			#		for _, _, text in paragraph.render(render_node):
			#			print("\t" + text)
			#print("\t\"\"\"")
			
			#print("\tpass")
			print()
		elif name in plains:
			print()
			
			if parent is None:
				print(f"class {name}:")
			else:
				print(f"class {name}({parent}):")
			
			for value in attributes[classname][Kind.slot]:
				print(f"\t{value}:object")
			for value in attributes[classname][Kind.property_]:
				print(f"\t{value}:object")
			for value in attributes[classname][Kind.constructor]:
				print("\t@classmethod")
				print(f"\tdef {value}(cls):")
				print("\t\tpass")
				print("\t")
			for value in attributes[classname][Kind.abstract_method]:
				print(f"\tdef {value}(self):")
				print("\t\tpass")
				print("\t")
			for value in attributes[classname][Kind.method]:
				print(f"\tdef {value}(self):")
				print("\t\tpass")
				print("\t")

			#print("\t\"\"\"")
			#for id_ in plains[name]:
			#	clause = specification.find_clause(id_)
			#	for paragraph in clause.paragraphs:
			#		for _, _, text in paragraph.render(render_node):
			#			print("\t" + text)			
			#print("\t\"\"\"")
			
			#print("\tpass")
			print()
		else:
			raise ValueError(f"{name} is an unknown type")
		
		for value in type_hierarchy[name]:
			generate_type(value, name if name not in unions else None)
		
		if name in unions:
			print()
			print(name + " = " + " | ".join(unions[name]))
			print()
	
	for key in reversed(sorted(toplevel)):
		generate_type(key, None)
	
	
	#print()
	#for name, subtypes in unions.items():
	#	if name in classes:
	#		continue
	#	print(name, "=", " | ".join(subtypes))
	#print()
	#for name, ids in classes.items():
	#	print(name, ids)
	#print()

	#print(types.keys())
	#print(enums.keys())
	#print(unions.keys())

quit()








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


def specification_abstract_operations(specification, restrict_to=None):
	"Prepare the spec of abstract operations. Yields all operations from the spec one by one."
	
	for clause in specification.find_clauses_with_title_ending_with(")"):
		title = clause.title
		if title.startswith("Static Semantics: "):
			title = title[18:]
		if title.startswith("Runtime Semantics: "):
			title = title[19:]
		if ("::" in title) or ("." in title) or ("%" in title) or (" Operator " in title) or ("[[" in title):
			continue
		
		if title.split(" ")[0] == 'CreateIntrinsics':
			continue
		
		if (restrict_to is not None) and all(not title.split(" ")[0] == _name for _name in restrict_to):
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
		

def specification_syntax_directed_operations(specification, restrict_to=None):
	"Prepare the spec of syntax-directed operations. Yields all operations from the spec. First yields the operation header, then all its subsections one by one, then the header of the next operation and so on."
	
	for clause in chain(specification.find_clauses_with_title_starting_with("Static Semantics:"), specification.find_clauses_with_title_starting_with("Runtime Semantics:")):
		if clause.title.endswith(": SV") or clause.title.endswith(": MV") or clause.title.endswith(": TV") or clause.title.endswith(": TRV"):
			continue
		if (restrict_to is not None) and all(not clause.title.startswith(_name + " ") for _name in restrict_to):
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
			t = ast_parse(code)
			if len(t.body) != 1 or t.body[0].__class__.__name__ != 'FunctionDef':
				raise ValueError
		except (SyntaxError, ValueError) as error:
			errors.append(error)
			try:
				spec += prompt[f"Algorithm / Syntax Error {len(errors)}"]
			except KeyError:
				pass
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def generate_evaluation(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate Evaluation."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Abstract Operation"] + prompt["Evaluation"]
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			t = ast_parse(code)
			if len(t.body) != 1 or t.body[0].__class__.__name__ != 'FunctionDef':
				raise ValueError
		except (SyntaxError, ValueError) as error:
			errors.append(error)
			try:
				spec += prompt[f"Algorithm / Syntax Error {len(errors)}"]
			except KeyError:
				pass
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def generate_condition(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate a single subsection of a syntax-directed operation."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Syntax Directed Operation"] + prompt["Abstract Operation"] + func_prompt.get(func_name, "")
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			t = ast_parse(code)
			if len(t.body) != 1 or t.body[0].__class__.__name__ != 'FunctionDef':
				raise ValueError
		except (SyntaxError, ValueError) as error:
			errors.append(error)
			try:
				spec += prompt[f"Algorithm / Syntax Error {len(errors)}"]
			except KeyError:
				pass
		else:
			return code
	else:
		raise ExceptionGroup("Model makes too many syntax errors!", errors)


def generate_algorithm(codegen, prompt, spec, func_name, func_args, func_arg_types, func_arg_optional, func_return_type):
	"Ask AI to generate a single abstract operation."
	personality = prompt[""] + prompt["Algorithm"] + prompt["Abstract Operation"] + func_prompt.get(func_name, "")
	prototype = 'def ' + func_name + '(' + ', '.join(_arg + ': ' + _type + (' = None' if _optional else '') for (_arg, _type, _optional) in zip(func_args, func_arg_types, func_arg_optional)) + ')' + ((' -> ' + func_return_type) if func_return_type else '') + ':\n\t'
	
	errors = []
	retries = max_syntax_retries
	while retries:
		retries -= 1
		code = codegen.request(personality, spec, prototype)
		try:
			t = ast_parse(code)
			if len(t.body) != 1 or t.body[0].__class__.__name__ != 'FunctionDef':
				raise ValueError
		except (SyntaxError, ValueError) as error:
			errors.append(error)
			print(code)
			try:
				spec += prompt[f"Algorithm / Syntax Error {len(errors)}"]
			except KeyError:
				pass
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


def compile_abstract_operations(specification, codegen, prompt, restrict_to=None):
	yield "#!/usr/bin/python3"
	yield ""
	yield ""
	yield "from definitions import *"
	yield ""
	yield ""
	for stype, spec in specification_abstract_operations(specification, restrict_to):
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
		yield generate_early_errors(codegen, prompt, spec, name, ('self', 'goal'), ('Self', 'Nonterminal'), (False, False), 'List[Error["SyntaxError"]]')
		final_function.append(f"\tif (errors := {name}(self, goal)) is not NotImplemented: return errors")
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


def compile_syntax_directed_operations(specification, codegen, prompt, restrict_to=None):
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
	for spec in specification_syntax_directed_operations(specification, restrict_to):
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


'''
if __debug__ and __name__ == '__main__':
	from pickle import load
	
	with Path('specification.pickle').open('rb') as fd:
		specification = load(fd)
	
	for id_ in specification.subclause_ids():
		clause = specification.find_clause(id_)
		if clause.paragraphs and isinstance(clause.paragraphs[0], Paragraph):
			text = " ".join(str(_token) for _token in clause.paragraphs[0].commands[0])
			print(((clause.title + ": ") if clause.title else "") + text)
	
	codegen = Codegen(url=environ['LLM_API_URL'], api_key=environ['LLM_API_KEY'])
	codegen.configure(**eval(environ['LLM_CONFIG_EXTRA']))
	codegen.configure(model=environ['LLM_MODEL'], rate_limit=4, min_delay=3, rate_limit_down=0.05, rate_limit_up=0.75)
	
	result = codegen.request(personality, spec, '{"__kind":').strip()

	quit()
'''




if __name__ == '__main__':
	from codegen import Codegen
	from lxml.etree import fromstring as xml_frombytes, tostring as xml_tobytes, tounicode as xml_tounicode
	from os import environ
	
	
	pickle_dir = Path('pickle')
	pickle_dir.mkdir(exist_ok=True)
	
	with (pickle_dir / 'specification.pickle').open('rb') as fd:
		specification = load(fd)
	
	print("URL:  ", environ['LLM_API_URL'])
	print("model:", environ['LLM_MODEL'])
	
	codegen = Codegen(url=environ['LLM_API_URL'], api_key=environ['LLM_API_KEY'])
	codegen.configure(**eval(environ['LLM_CONFIG_EXTRA']))
	codegen.configure(model=environ['LLM_MODEL'], rate_limit=4, min_delay=3, rate_limit_down=0.05, rate_limit_up=0.75)
	




	language_types = set()
	specification_types = set()
	type_definitions = defaultdict(lambda: {'ids':[], 'superclass':None})
	
	for section_number in range(20, 21):
		categories_file = pickle_dir / f'categories_{section_number}.pickle'
		
		try:
			with categories_file.open('rb') as fd:
				categories = load(fd)
		except IOError:
			try:
				categories_file.unlink()
			except IOError:
				pass
			categories = {}
		
		
		if section_number >= 6:
			for id_, category_seq in categories.items():
				if not isinstance(category_seq, tuple):
					category_seq = [category_seq]
				for category in category_seq:
					if category['category'] != 'type': continue
					if category['kind'] == 'unknown': continue
					
					try:
						if category['specification type']:
							specification_types.add(category['name'])
					except KeyError:
						pass
					
					#print(category)
					type_definitions[category['name']]['ids'].append(id_)
					type_definitions[category['name']].update({_key:_value for _key, _value in category.items() if _key != 'name'})
					#if type_definitions[category['name']]['union']
		
		
		
		#quit()
		
		'''
		try:
			do = False
			for nn, id_ in enumerate(specification.subclause_ids()):
				#if id_ in categories: continue
				
				clause = specification.find_clause(id_)
				if clause.chapter:
					do = clause.chapter[0] == section_number
				if not do: continue
				
				if "Set." in clause.title:
					pass
				elif clause.title.startswith("Set "):
					pass
				elif "%" in clause.title:
					pass
				elif any(_rw in clause.title for _rw in ["return", "catch", "finally", "try", "next"]):
					pass
				else:
					continue
				#if '::' in clause.title: continue
				#if ')' in clause.title: continue
				
				#try:
				#	clause.paragraphs[0].commands[0]
				#except IndexError:
				#	continue
				
				text = ""
				try:
					for cn in range(2):
						text += " ".join(str(_token) for _token in clause.paragraphs[0].commands[cn]) + " "
				except (IndexError, AttributeError):
					text += " (nothing more)."
				
				spec = ((clause.title + ": ") if clause.title else "") + text
				
				#if "A specification type" not in spec: continue
				#if "Number" not in spec: continue
				
				print(clause.chapter, clause.title, "////", spec)
				personality = prompt[""] + prompt["Guess content"]
				
				retries = 3
				while retries:
					retries -= 1
					result = codegen.request(personality, spec, '{"category":').strip()
					print(result)
					
					try:
						result_val = literal_eval(result)
						categories[clause.id_] = result_val
						
						if not isinstance(result_val, tuple):
							result_val = result_val,
						for result_dict in result_val:
							if result_dict['category'] in ['informal', 'normative']:
								pass
							else:
								result_dict['kind']
								if result_dict['name'].startswith("ECMAScript"):
									raise ValueError("name must not start with ECMAScript")
					except (SyntaxError, ValueError, KeyError) as error:
						try:
							del categories[clause.id_]
						except KeyError:
							pass
						print("Error", error)
					else:
						break
				
				print()
		
		finally:
			with categories_file.open('wb') as fd:
				dump(categories, fd)
		'''


	if 'SpecificationType' in type_definitions:
		type_definitions['SpecificationType']['union'] = list(specification_types)
	#print()
	for type_name, props in type_definitions.items():
		if props['kind'] == 'union': continue
		if props['superclass'] is not None: continue
		print(type_name, props)
	for type_name, props in type_definitions.items():
		if props['superclass'] is None: continue
		print(type_name, props)
	for type_name, props in type_definitions.items():
		if props['kind'] != 'union': continue
		print(type_name, props)
	
	print("Functions:")
	for id_, category_seq in categories.items():
		if not isinstance(category_seq, tuple):
			category_seq = [category_seq]
		for category in category_seq:
			if category['category'] == 'function':
				print(category['name'], category['kind'], "\t\t", id_)
	print()

	
	quit()
	
	dest_dir = Path('gencode')
	dest_dir.mkdir(exist_ok=True)
	
	problematic_functions = None
	#problematic_functions = ["ParseText"]
	#problematic_functions += ["Set", "PutValue", "CreateArrayIterator", "DoWait", "NewPromiseReactionJob", "MinFromTime", "TimeWithinDay", "Day", "GetValueFromBuffer", "SetValueInBuffer", "GetRawBytesFromSharedBlock"]
	#problematic_functions += ["EvaluateNew", "InitializeReferencedBinding", "NumberBitwiseOp"]
	#problematic_functions += ["MakeTime", "YearFromTime"]
	
	verbose = True
	tee_files(dest_dir / 'ao_library.py', print_=verbose)(compile_abstract_operations)(specification, codegen, prompt, problematic_functions)
	#tee_files(dest_dir / 'early_errors.py', print_=verbose)(compile_early_errors)(specification, codegen, prompt)
	#tee_files(dest_dir / 'evaluation.py', print_=verbose)(compile_evaluation)(specification, codegen, prompt)
	#tee_files(dest_dir / 'sdo_library.py', print_=verbose)(compile_syntax_directed_operations)(specification, codegen, prompt, problematic_functions)

