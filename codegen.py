#!/usr/bin/python3


import requests
import json
from time import sleep
from random import choice, randint
from textwrap import dedent


class MockCodegen:
	"Mock code generator that creates random gibberish."
	
	def __init__(self, **kwargs):
		pass
	
	def configure(self, **kwargs):
		pass
	
	def list_models(self):
		return []
	
	def __random_str(self):
		result = ""
		while ":" not in result:
			result = "".join(choice(":::  _abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _n in range(randint(40, 95)))
		return result
	
	def request(self, system, user, prefix=None):
		# TODO: generate 
		if prefix is None:
			prefix = ""
		return prefix + "\n\t".join(self.__random_str() for _n in range(randint(5, 15)))


class Codegen:
	def __init__(self, url, api_key=None):
		self.__url = url
		self.__api_key = api_key
		
		self.model = None
		self.temperature = 0.1
		self.connect_timeout = 3.1
		self.read_timeout = 27.1
		
		self.retries = 4
		self.min_delay = 0
		self.max_delay = 15
		self.rate_limit = 3.5
		self.rate_limit_down = 0.01
		self.rate_limit_up = 0.1
		
		self.supports_listing = True
		self.prepend_prefix = False
		self.skip_prefix_param = False
		self.replace_escapes = False
		self.dedent = False
		#self.extra = {}
	
	def list_models(self):
		if not self.supports_listing:
			return
		
		url = self.__url + '/models'
		
		headers = {'Accept': 'application/json'}
		if self.__api_key is not None:
			headers['Authorization'] = "Bearer " + self.__api_key
		response = requests.get(url, headers=headers, timeout=(self.connect_timeout, self.read_timeout))
		response.raise_for_status()
		j = response.json()
		for model in j['data']:
			yield model['id']
	
	def configure(self, **kwargs):
		for key, value in kwargs.items():
			if not hasattr(self, key):
				raise AttributeError(f"Key not found: `{key}`")
			setattr(self, key, value)
	
	def raw_request(self, persona, spec, prefix=None):
		url = self.__url + '/chat/completions'

		if prefix is not None:
			assistant = {'role': "assistant"}
			if not self.skip_prefix_param:
				assistant['prefix'] = True
			assistant['content'] = prefix
		else:
			assistant = None
		
		data = {
			'model': self.model,
			'temperature': self.temperature,
			'messages': [
				{'role': "system", 'content': persona},
				{'role': "user", 'content': spec}
			] + ([assistant] if assistant is not None else [])
		}
		#data.update(self.extra)
		headers = {'Content-type': 'application/json'}
		if self.__api_key is not None:
			headers['Authorization'] = "Bearer " + self.__api_key
		
		#print(json.dumps(data))
		response = requests.post(url, data=json.dumps(data), headers=headers, timeout=(self.connect_timeout, self.read_timeout))
		
		try:
			response.raise_for_status()
		except Exception as error:
			print("error", response.text)
			raise
		
		result = response.json()
		content = result['choices'][0]['message']['content']
		prompt_tokens = result['usage']['prompt_tokens']
		completion_tokens = result['usage']['completion_tokens']
		return content, prompt_tokens, completion_tokens
	
	def request(self, persona, spec, prefix=None):
		if self.model is None:
			raise ValueError("Configure model first.")
		retry = self.retries
		exceptions = []
		while retry > 0:
			retry -= 1
			if self.rate_limit > self.max_delay: self.rate_limit = self.max_delay
			if self.rate_limit < self.min_delay: self.rate_limit = self.min_delay
			if self.rate_limit > 0:
				print(f"Sleeping for {self.rate_limit:.2f}s.")
				sleep(self.rate_limit)
			
			try:
				print("Making request.")
				response, prompt_tokens, completion_tokens = self.raw_request(persona, spec, prefix)
			except Exception as error:
				print(type(error).__name__ + ":", error, ".")
				exceptions.append(error)
				self.rate_limit += self.rate_limit_up
			else:
				print("response length:", len(response))
				self.rate_limit -= self.rate_limit_down
				if response.startswith("```python\n"):
					end = response.find("```", 10)
					result = response[10:end]
				elif response.startswith("```\n"):
					if response.endswith("```"):
						result = response[4:-3]
					elif response.endswith("`"):
						result = response[4:-1]
					else:
						result = response[4:]
				elif response.startswith("```"):
					if response.endswith("```"):
						result = response[3:-3]
					elif response.endswith("`"):
						result = response[3:-1]
					else:
						result = response[3:]
				elif response.startswith("`"):
					if response.endswith("`</s>"):
						result = response[1:-5]
					elif response.endswith("`"):
						result = response[1:-1]
					else:
						result = response[1:]
				else:
					result = response
				
				#result = result.lstrip()
				
				if (prefix is not None) and self.prepend_prefix:
					result = prefix + result
				
				if self.replace_escapes:
					result = result.replace('\\_', '_')
				
				if self.dedent:
					result = dedent(result)
				
				return result
		
		raise ExceptionGroup("All retries failed.", exceptions)


if __name__ == '__main__':
	from os import environ
	
	print("URL:  ", environ['LLM_API_URL'])
	print("model:", environ['LLM_MODEL'])
	
	codegen = Codegen(url=environ['LLM_API_URL'], api_key=environ['LLM_API_KEY'])
	codegen.configure(**eval(environ['LLM_CONFIG_EXTRA']))
	print(list(codegen.list_models()))
	codegen.configure(model=environ['LLM_MODEL'])
	
	print()
	print()
	print("Test 0: test")
	print()
	result = codegen.request("This is a test.", "Generate this string: 'hello void' (nothing else).", "'h")
	print(result)
	print()
	
	personality = """
		You are a code generator. You print functions in Python language, as specified by requirements. Print only blocks of code.
		Do not chat outside code blocks. Do not provide explanations nor usage examples. Keep identifiers as they are, do not change case or convert to snake case, but append underscore if the identifier is a reserved word.
		Use the tab character for indentation.
	"""
	
	print()
	print()
	print("Test 1: prime")
	print()
	result = codegen.request(personality, "A function that makes a Roman numeral from the given integer.", 'def roman(n:int)->str:\n\t')
	print(result)
	print()
	
	print()
	print()
	print("Test 2: relatively prime")
	print()
	result = codegen.request(personality, "A function that tests if the numbers `m` and `n` are relatively prime.", 'def relatively_prime(m:int, n:int)->bool:\n\t')
	print(result)
	print()
	
	print()
	print()
	print("Test 3: maximum")
	print()
	result = codegen.request(personality, "A function of two arguments that returns the argument with the greatest absolute value.", 'def greater_abs(m:int, n:int)->int:\n\t')
	print(result)
	print()
	
	print()
	print()
	print("Test 4: ECMA algorithm (1)")
	print()
	spec_hints = "Whenever the specification says to take FIELD1 of FIELD2, take it as `self.get_subtree('FIELD2').FIELD1()`. If a variable name is `list` convert it to `list_`."
	
	prototype = 'PropertyNameList_1(self:ParseNode) -> list[str]'
	spec = """
    	(1.) Let propName be the PropName of PropertyDefinition.
		(2.) If propName is empty, return a new empty List.
		(3.) Return « propName ».
	"""
	result = codegen.request(personality + spec_hints, spec, f"def {prototype}:\n\t")
	print(result)
	print()
	
	print()
	print()
	print("Test 5: ECMA algorithm (2)")
	print()
	prototype = 'PropertyNameList_2(self:ParseNode) -> list[str]'
	spec = """
		(1.) Let list be the PropertyNameList of PropertyDefinitionList.
		(2.) Let propName be the PropName of PropertyDefinition.
		(3.) If propName is empty, return list.
		(4.) Return the list-concatenation of list and « propName ».
	"""
	result = codegen.request(personality + spec_hints, spec, f"def {prototype}:\n\t")
	print(result)
	print()
	
	print()
	print()
	print("Test 6: linear regression")
	print()
	prototype = 'linear_regression(values:list[float]) -> tuple[float, float]'
	spec = """
		A function that calculates linear regression of the provided values list.
		The argument `values` is interpreted as values of a mathematical function `f` at points 0, 1, 2, ...
		The program should return two floats `a` and `b` such that the function `a * x + b` approximates the function `f`, minimizing mean squared deviation.
		The first parameter `a` is the slope, the second `b` is the intercept.
	"""
	result = codegen.request(personality, spec, f"def {prototype}:\n\t")
	print(result)
	print()
	
	print()
	print()
	print("Test 7: EBNF grammar")
	print()
	prototype = 'match_prefix(self, source:list[str], position:int, goal:Nonterminal) -> tuple[ParseNode, int]'
	spec = """
		Make a parser of EBNF grammar.

```
@dataclass
class Nonterminal:
	def __init__(self, value):
		self.value = value

ANY_TOKEN = Nonterminal('*')


@dataclass
class Terminal:
	def __init__(self, value):
		self.value = value


class Options(Flag):
	standard = 0
	optional = 1


class Production:
	\"\"\"
	The class `Production` represents a single production of EBNF grammar.
	The field `head` is a `Nonterminal` to expand, that is the left-hand side of the production. There may be many `Production`s with the same `head` in the grammar.
	The field `symbols` is a list of `Terminal`s and `Nonterminal`s, that is the right-hand side of the production.
	The field `options` is a list of `Options` elements. It should be of equal length to `symbols`. Options modify expansion of the corresponding `symbols`.
	\"\"\"
	def __init__(self:Self, head:Nonterminal, symbols:list[Terminal|Nonterminal], options:list[Options]=None):
		self.head = head
		self.symbols = symbols
		if options is None:
			options = [Options.standard] * len(symbols)
		self.options = options


class ParseNode:
	\"\"\"
	The class `ParseNode` represents a node of Abstract Syntax Tree. Its field `head` represents the production this subtree was produced from.
	The field `symbols` is a list of `Terminal`s and `Nonterminal`s representing the right-hand side of the production.
	The field `children` is a list of children that may be a subtree (another `ParseNode`), a string or None.
	In general, if the n-th element of `symbols` is a `Nonterminal`, then the corresponding n-th element of `children` will be `ParseNode` with the same `Nonterminal` in `head`,
	however it may be also None if the corresponding `Nonterminal` was optional.
	If the n-th element of `symbols` is a `Terminal`, then the corresponding n-th element of `children` will be a string equal to the `Terminal`s `value` field,
	however it may be also None if the corresponding `Terminal` was optional.
	\"\"\"
	
	def __init__(self, head:Nonterminal, symbols:list[Terminal|Nonterminal], children:list[Self|str|None]):
		self.head = head
		self.symbols = symbols
		self.children = children
		
		assert len(children) == len(symbols)
		assert all(isinstance(_symbol, Terminal | Nonterminal) for _symbol in symbols)
		if __debug__:
			for child, symbol in zip(children, symbols):
				if isinstance(symbol, Nonterminal):
					assert child is None or isinstance(child, self.__class__)
				elif isinstance(symbol, Terminal):
					assert child is None or isinstance(child, str)
				else:
					raise ValueError

class Grammar:
	"EBNF grammar."
	
	def __init__(self):
		self.productions = defaultdict(list)
	
	def add_production(self, production:Production):
		"Add the production to the grammar."
		self.productions[production.head].append(production)
	
	def match_prefix(self, source:list[str], position:int, goal:Nonterminal) -> tuple[ParseNode, int]:
		...

```
	
	Finish the method `match_prefix` that is an LR parser.
	
	The argument `source` is a tokenized source, a list of strings.
	The argument `position` is the position where matching should happen. The argument `goal` is the `Nonterminal` we're trying to match.
	The field `self.productions` contains the productions of the grammar.
	The grammar may be left-recursive, meaning a `Nonterminal(XXX) may appear as the leftmost symbol of the production XXX.
	
	If the argument `position` is lesser than 0 or greater or equal to `source` length, raise ValueError.
	If a match is found, the method should return `ParseTree` representing the abstract syntax tree and a number of tokens from `source` that were consumed.
	If no match is found, the method should return (None, 0).
	A special nonterminal `ANY_TOKEN = Nonterminal('*')` should match any single token.
	
	"""
	result = codegen.request(personality, spec, f"def {prototype}:\n\t")
	print(result)
	print()

