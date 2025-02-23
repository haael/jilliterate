#!/usr/bin/python3


__all__ = 'Cmd', 'Note', 'Paragraph', 'Table', 'Algorithm', 'Clause', 'Specification'


from lxml.etree import fromstring as xml_frombytes, tostring as xml_tobytes, tounicode as xml_tounicode
from enum import Enum, Flag
from collections import defaultdict, Counter, deque
from itertools import chain
from typing import Self
from dataclasses import dataclass
from random import choices

from datatypes import *
from tokens import *


class Cmd(TypedRecord):
	content: list_of(Token | str | Grammar | SpecialSymbol)
	datum: object
	
	def __init__(self, content=None, datum=None):
		if content is not None:
			self.content = list(content)
		else:
			self.content = []
		
		self.datum = datum
		
		if __debug__: self._validate()
	
	def _print(self, level=0):
		yield level, " ".join(_t if isinstance(_t, str) else repr(_t) for _t in self.content)
		if self.datum:
			yield from self.datum._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Cmd',)
		tokens = (token if isinstance(token, str|SpecialSymbol) else token.render(fn, path, index + (n,)) for (n, token) in enumerate(self.content))
		if self.datum is not None:
			d = self.datum.render(fn, path, index)
		else:
			d = None
		#print(fn)
		yield from fn(self, path, index, content=tokens, datum=d)
	
	constants = FilteringProperty('content', Constant)
	variables = FilteringProperty('content', Variable)
	terminals = FilteringProperty('content', Terminal)
	nonterminals = FilteringProperty('content', Nonterminal)
	enums = FilteringProperty('content', Enum_)
	fields = FilteringProperty('content', ObjectField)
	definitions = FilteringProperty('content', Definition)
	grammars = FilteringProperty('content', Grammar)
	
	def find_definition(self, id_):
		for definition in self.definitions:
			#print("Definition in", self, definition, "match", id_)
			if definition.id_ == id_:
				return definition
		else:
			raise KeyError(f"Definition with id `{id_}` not found.")
	
	def __iter__(self):
		yield from self.content


class Paragraph(TypedRecord):
	commands: list_of(Cmd)
	
	def __init__(self, commands=None):
		if commands is not None:
			self.commands = list(commands)
		else:
			self.commands = []
		
		if __debug__:
			self._validate()
			for child in self.commands:
				child._validate()
	
	def _print(self, level=0):
		yield level, "*"
		for command in self.commands:
			yield from command._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Paragraph',)
		commands = (command.render(fn, path, index + (n,)) for (n, command) in enumerate(self.commands))
		yield from fn(self, path, index, commands=commands)
	
	def __bool__(self):
		return bool(self.commands)
	
	constants = GatheringProperty('constants', ['commands'])
	variables = GatheringProperty('variables', ['commands'])
	terminals = GatheringProperty('terminals', ['commands'])
	nonterminals = GatheringProperty('nonterminals', ['commands'])
	enums = GatheringProperty('enums', ['commands'])
	fields = GatheringProperty('fields', ['commands'])
	definitions = GatheringProperty('definitions', ['commands'])
	grammars = GatheringProperty('grammars', ['commands'])
	
	def find_definition(self, id_):
		for cmd in self.commands:
			try:
				return cmd.find_definition(id_)
			except KeyError:
				pass
		else:
			raise KeyError(f"Definition with id `{id_}` not found.")


class Note(TypedRecord):
	title: str
	commands: list_of(Cmd)
	
	def _print(self, level=0):
		yield level, "* " + self.title + ":"
		for command in self.commands:
			yield from command._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Note',)
		commands = (command.render(fn, path, index + (n,)) for (n, command) in enumerate(self.commands))
		yield from fn(self, path, index, commands=commands)


class Algorithm(TypedRecord):
	instructions: list_of(Cmd | Self)
	
	def __init__(self):
		self.instructions = []
		if __debug__:
			self._validate()
			for child in self.instructions:
				child._validate()
	
	def _print(self, level=0):
		yield level, "* Algorithm:"
		for instruction in self.instructions:
			yield from instruction._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Algorithm',)
		instructions = (instruction.render(fn, path, index + (n,)) for (n, instruction) in enumerate(self.instructions))
		yield from fn(self, path, index, instructions=instructions)
	
	def __bool__(self):
		return bool(self.instructions)
	
	constants = GatheringProperty('constants', ['instructions'])
	variables = GatheringProperty('variables', ['instructions'])
	terminals = GatheringProperty('terminals', ['instructions'])
	nonterminals = GatheringProperty('nonterminals', ['instructions'])
	enums = GatheringProperty('enums', ['instructions'])
	fields = GatheringProperty('fields', ['instructions'])
	definitions = GatheringProperty('definitions', ['instructions'])
	grammars = GatheringProperty('grammars', ['instructions'])
	
	def find_definition(self, id_):
		for instruction in self.instructions:
			try:
				return instruction.find_definition(id_)
			except KeyError:
				pass
		else:
			raise KeyError(f"Definition with id `{id_}` not found.")


class Table(TypedRecord):
	id_: str
	head: list_of(list_of(Paragraph|Algorithm))
	body: list_of(list_of(Paragraph|Algorithm))
	
	def __init__(self, id_):
		self.id_ = id_
		self.head = []
		self.body = []
		if __debug__:
			self._validate()
			for row in self.head:
				row._validate()
			for row in self.body:
				row._validate()
	
	def render(self, fn, path=(), index=()):
		path = path + ('Table',)
		head = ((col.render(fn, path, index + (m, n)) for (n, col) in row) for (m, row) in enumerate(self.head))
		body = ((col.render(fn, path, index + (m, n)) for (n, col) in row) for (m, row) in enumerate(self.body))
		yield from fn(self, path, index, head=head, body=body)
	
	constants = GatheringProperty('constants', ['head', 'body'])
	variables = GatheringProperty('variables', ['head', 'body'])
	terminals = GatheringProperty('terminals', ['head', 'body'])
	nonterminals = GatheringProperty('nonterminals', ['head', 'body'])
	enums = GatheringProperty('enums', ['head', 'body'])
	fields = GatheringProperty('fields', ['head', 'body'])
	definitions = GatheringProperty('definitions', ['head', 'body'])
	grammars = GatheringProperty('grammars', ['head', 'body'])
	
	def find_definition(self, id_):
		if self.id_ == id_:
			return self
		
		for row in self.body:
			for cmd in row:
				try:
					return cmd.find_definition(id_)
				except KeyError:
					pass
		else:
			raise KeyError(f"Definition with id `{id_}` not found.")
	
	def _print(self, level=0):
		yield level, "======"
		
		for n, row in enumerate(self.head):
			for m, cmd in enumerate(row):
				if m:
					yield level, "------"
				yield from (_v for (_l, _v) in enumerate(cmd._print(level)) if _l)
			yield level, "==--=="
		
		for n, row in enumerate(self.body):
			for m, cmd in enumerate(row):
				if m:
					yield level, "------"
				yield from (_v for (_l, _v) in enumerate(cmd._print(level)) if _l)
			yield level, "======"


class Clause(TypedRecord):
	id_: str
	chapter: tuple_of(int)
	title: str
	paragraphs: list_of(Paragraph | Note | Table | Grammar | Algorithm)
	subclauses: list_of(Self)
	
	def __init__(self):
		self.id_ = ''
		self.chapter = ()
		self.title = ""
		self.paragraphs = []
		self.subclauses = []
		if __debug__:
			self._validate()
			for child in self.paragraphs:
				child._validate()
			for child in self.subclauses:
				child._validate()
	
	def _print(self, level=0):
		yield level, "* Clause " + ".".join(str(_d) for _d in self.chapter) + " " + repr(self.title) + " [" + self.id_ + "]:"
		for paragraph in self.paragraphs:
			yield from paragraph._print(level + 1)
		for clause in self.subclauses:
			yield from clause._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Clause',)
		paragraphs = (paragraph.render(fn, path, index + (n,)) for (n, paragraph) in enumerate(self.paragraphs))
		subclauses = (subclause.render(fn, path, index + (n,)) for (n, subclause) in enumerate(self.subclauses))
		yield from fn(self, path, index, paragraphs=paragraphs, subclauses=subclauses)
	
	def subclause_ids(self):
		yield self.id_
		for subclause in self.subclauses:
			yield from subclause.subclause_ids()
	
	def find_clause(self, id_):
		if self.id_ == id_:
			return self
		else:
			for subclause in self.subclauses:
				try:
					return subclause.find_clause(id_)
				except KeyError:
					pass
			else:
				raise KeyError(f"Clause with id `{id_}` not found.")
	
	def find_definition(self, id_):
		if self.id_ == id_:
			return self
		
		for paragraph in self.paragraphs:
			if hasattr(paragraph, 'id_') and paragraph.id_ == id_:
				return paragraph
			
			if not hasattr(paragraph, 'find_definition'):
				continue
			
			try:
				return paragraph.find_definition(id_)
			except KeyError:
				pass
		
		for subclause in self.subclauses:
			try:
				return subclause.find_definition(id_)
			except KeyError:
				pass
		
		raise KeyError(f"Definition with id `{id_}` not found.")
	
	def find_clause_with_definition(self, id_):
		for paragraph in self.paragraphs:
			if not hasattr(paragraph, 'find_definition'):
				continue
			
			try:
				paragraph.find_definition(id_)
			except KeyError:
				pass
			else:
				return self
		
		for subclause in self.subclauses:
			try:
				return subclause.find_clause_with_definition(id_)
			except KeyError:
				pass
		
		raise KeyError(f"Definition with id `{id_}` not found.")
	
	def find_clause_with_chapter(self, chapter):
		#print("clause", self.chapter, chapter)
		if not self.chapter:
			raise KeyError
		elif self.chapter == chapter:
			return self
		elif len(chapter) > len(self.chapter) and self.chapter == chapter[:len(self.chapter)]:
			for subclause in self.subclauses:
				try:
					return subclause.find_clause_with_chapter(chapter)
				except KeyError:
					pass
			else:
				raise KeyError
		else:
			raise KeyError
	
	def find_clauses_with_title(self, title):
		if self.title == title:
			yield self
		for subclause in self.subclauses:
			yield from subclause.find_clauses_with_title(title)
	
	def find_clauses_with_title_starting_with(self, title):
		if self.title.startswith(title):
			yield self
		for subclause in self.subclauses:
			yield from subclause.find_clauses_with_title_starting_with(title)
	
	def find_clauses_with_title_ending_with(self, title):
		if self.title.endswith(title):
			yield self
		for subclause in self.subclauses:
			yield from subclause.find_clauses_with_title_ending_with(title)
	
	constants = GatheringProperty('constants', ['paragraphs', 'subclauses'])
	variables = GatheringProperty('variables', ['paragraphs', 'subclauses'])
	terminals = GatheringProperty('terminals', ['paragraphs', 'subclauses'])
	nonterminals = GatheringProperty('nonterminals', ['paragraphs', 'subclauses'])
	enums = GatheringProperty('enums', ['paragraphs', 'subclauses'])
	fields = GatheringProperty('fields', ['paragraphs', 'subclauses'])
	definitions = GatheringProperty('definitions', ['paragraphs', 'subclauses'])
	
	@property
	def grammars(self):
		for paragraph in self.paragraphs:
			if isinstance(paragraph, Grammar):
				yield paragraph
			elif isinstance(paragraph, Algorithm):
				pass
			else:
				yield from paragraph.grammars
		
		for subclause in self.subclauses:
			yield from subclause.grammars


class Specification:
	"Parser of ECMA specifications in XML format."
	
	emu_namespace = 'https://github.com/rbuckton/grammarkdown'
	html_namespace = 'http://www.w3.org/1999/xhtml'
	
	def __init__(self):
		self.toplevel_clauses = []
		#self.chapter_cache = {} # FIXME
		#self.grammars = {}
	
	def _print(self, level=0):
		yield level, "* ECMA Specification:"
		for clause in self.toplevel_clauses:
			yield from clause._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Specification',)
		yield from fn(self, path, index, toplevel_clauses=(clause.render(fn, path, index + (n,)) for (n, clause) in enumerate(self.toplevel_clauses)))
	
	def parse(self, xml):
		if xml.tag == f'{{{self.emu_namespace}}}clause':
			clause = self.clause(xml)
			self.toplevel_clauses.append(clause)
			
			if __debug__:
				for subclause in self.toplevel_clauses:
					subclause._validate()
			
			return clause
		
		elif xml.tag == f'{{{self.emu_namespace}}}intro' or xml.tag == f'{{{self.emu_namespace}}}annex':
			return None
		
		else:
			raise ValueError(f"Expected <emu:clause/> or <emu:intro/> got {xml.tag}.")
	
	def subclause_ids(self):
		for clause in self.toplevel_clauses:
			yield from clause.subclause_ids()
	
	def find_clause(self, id_):
		for clause in self.toplevel_clauses:
			try:
				return clause.find_clause(id_)
			except KeyError:
				pass
		else:
			raise KeyError(f"Clause with id `{id_}` not found.")
	
	def find_definition(self, id_):
		for clause in self.toplevel_clauses:
			try:
				return clause.find_definition(id_)
			except KeyError:
				pass
		else:
			raise KeyError(f"Definition with id `{id_}` not found.")
	
	def find_clause_with_definition(self, id_):
		for clause in self.toplevel_clauses:
			try:
				return clause.find_clause_with_definition(id_)
			except KeyError:
				pass
		else:
			raise KeyError(f"Definition with id `{id_}` not found.")
	
	def find_clauses_with_title(self, title):
		for clause in self.toplevel_clauses:
			yield from clause.find_clauses_with_title(title)
	
	def find_clause_with_chapter(self, chapter):
		for clause in self.toplevel_clauses:
			if clause.chapter == chapter:
				return clause
			elif len(chapter) > len(clause.chapter) and clause.chapter == chapter[:len(clause.chapter)]:
				return clause.find_clause_with_chapter(chapter)
		else:
			raise KeyError("Chapter " + ".".join(str(_d) for _d in chapter) + " not found.")
	
	#def find_clause_with_chapter(self, chapter):
	#	try:
	#		return self.chapter_cache[chapter]
	#	except KeyError:
	#		pass
	#	
	#	for id_ in self.subclause_ids():
	#		clause = self.find_clause(id_)
	#		if clause.chapter == chapter:
	#			self.chapter_cache[chapter] = clause
	#			return clause
	#	else:
	#		raise KeyError
	
	def find_clauses_with_title_starting_with(self, title):
		for clause in self.toplevel_clauses:
			yield from clause.find_clauses_with_title_starting_with(title)
	
	def find_clauses_with_title_ending_with(self, title):
		for clause in self.toplevel_clauses:
			yield from clause.find_clauses_with_title_ending_with(title)
	
	def find_ref(self, href):
		if not href.startswith('#'):
			raise ValueError("Href must start with #, got: " + href)
		
		id_ = href[1:]
		try:
			return self.find_clause(id_)
		except KeyError:
			pass
		
		try:
			return self.find_clause_with_definition(id_)
		except KeyError:
			pass
		
		raise KeyError(f"Ref {href} not found.")
		
	subclause_tags = frozenset({f'{{{emu_namespace}}}clause', f'{{{html_namespace}}}h2'})
	
	subparagraph_tags = frozenset({
		f'{{{html_namespace}}}p', f'{{{html_namespace}}}div', f'{{{html_namespace}}}span', f'{{{html_namespace}}}ul', f'{{{html_namespace}}}ol', f'{{{emu_namespace}}}note',
		f'{{{emu_namespace}}}alg', f'{{{html_namespace}}}table', f'{{{emu_namespace}}}table', f'{{{html_namespace}}}pre', f'{{{html_namespace}}}figure', f'{{{emu_namespace}}}import'
	})
	
	def clause(self, xml):
		if not hasattr(xml, 'tag'): raise ValueError("The value must be an XML element.")
		if xml.tag != f'{{{self.emu_namespace}}}clause': raise ValueError("The tag must be <emu:clause/>.")
		
		#print("clause", xml.attrib['id'], [_child.tag for _child in xml], any(_child.tag in self.subclause_tags for _child in xml))
		clause = Clause()
		
		try:
			clause.id_ = xml.attrib['id']
		except KeyError:
			pass
		
		#try:
		#	clause.type_ = xml.attrib['type']
		#except KeyError:
		#	pass
		
		#try:
		#	clause.aoid = xml.attrib['aoid']
		#except KeyError:
		#	pass
		
		try:
			h1 = [_child for _child in xml if _child.tag == f'{{{self.html_namespace}}}h1'][0]
		except IndexError:
			pass
		else:
			title = self.extract_text(h1)
			
			clause.chapter = tuple(map(int, title.split(' ')[0].split('.')))
			clause.title = ' '.join(title.split(' ')[1:])
			
			#print(clause.chapter)
			
			#args = []
			#for subchild in h1:
			#	if subchild.tag == f'{{{self.html_namespace}}}var':
			#		args.append(Variable(subchild.text.strip()))
			#clause.args = args
		
		if any(_child.tag in self.subclause_tags for _child in xml):
			k = 0
			subclause = None
			
			if xml.text:
				subclause = Clause()
				subclause.id_ = clause.id_ + '#' + str(k)
				k += 1
				clause.subclauses.append(subclause)
				paragraph = Paragraph(self.commands(self.tokenize_string(xml.text)))
				if paragraph:
					subclause.paragraphs.append()
			
			for n, child in enumerate(xml):
				if child.tag == f'{{{self.html_namespace}}}h2':
					subclause = Clause()
					subclause.id_ = clause.id_ + '#' + str(k)
					k += 1
					subclause.title = self.extract_text(child)
					clause.subclauses.append(subclause)
				
				elif child.tag == f'{{{self.emu_namespace}}}clause':
					clause.subclauses.append(self.clause(child))
					subclause = None
				
				else:
					if subclause is None:
						subclause = Clause()
						subclause.id_ = clause.id_ + '#' + str(k)
						k += 1
						clause.subclauses.append(subclause)
					subclause.paragraphs.extend(_paragraph for _paragraph in self.paragraphs(child) if _paragraph)
					if child.tail:
						paragraph = Paragraph(self.commands(self.tokenize_string(child.tail)))
						if paragraph:
							subclause.paragraphs.append()
		
		else:
			clause.paragraphs = list(self.paragraphs(xml))
		
		if __debug__: clause._validate()
		return clause
	
	def __subparagraph(self, xml, level):
		if xml.tag == f'{{{self.emu_namespace}}}grammar' and level == 0:
			return True
		
		if xml.tag in self.subparagraph_tags:
			#if xml.tag == f'{{{self.html_namespace}}}span' and xml.attrib.get('class', '') != 'note':
			#	return False
			return True
		
		return any(self.__subparagraph(_child, level + 1) for _child in xml)
	
	def paragraphs(self, xml, level=0):
		"May yield Paragraph, Algorithm, Table or Grammar."
		
		if xml.tag == f'{{{self.html_namespace}}}h1':
			pass
		
		elif xml.tag == f'{{{self.emu_namespace}}}alg':
			yield self.algorithm(xml)
		
		elif xml.tag == f'{{{self.emu_namespace}}}table':
			yield self.table(xml)
		
		elif xml.tag == f'{{{self.emu_namespace}}}grammar':
			yield self.grammar(xml)
		
		elif any(self.__subparagraph(_child, level) for _child in xml):
			if xml.tag == f'{{{self.emu_namespace}}}note':
				note = []
			else:
				note = None
			
			tokens = deque()
			
			if xml.text:
				tokens.extend(self.tokenize_string(xml.text))
			
			for child in xml:
				for m, subparagraph in enumerate(self.paragraphs(child, level + 1)):
					if m == 0:
						if isinstance(subparagraph, Paragraph) and not self.__subparagraph(child, level):
							tokens.extend(chain.from_iterable(_command.content for _command in subparagraph.commands))
						else:
							if tokens:
								paragraph = Paragraph(self.commands(tokens))
								if __debug__: paragraph._validate()
								if note is None:
									yield paragraph
								else:
									note.append(paragraph)
								tokens.clear()
							if note is None:
								yield subparagraph
							else:
								note.append(subparagraph)
					else:
						if tokens:
							paragraph = Paragraph(self.commands(tokens))
							if __debug__: paragraph._validate()
							if note is None:
								yield paragraph
							else:
								note.append(paragraph)
							tokens.clear()
						
						if isinstance(subparagraph, Paragraph) and not self.__subparagraph(child, level):
							tokens.extend(chain.from_iterable(_command.content for _command in subparagraph.commands))
						else:
							if note is None:
								yield subparagraph
							else:
								note.append(subparagraph)
				
				if child.tail:
					tokens.extend(self.tokenize_string(child.tail))
			
			if tokens:
				paragraph = Paragraph(self.commands(tokens))
				if __debug__: paragraph._validate()
				if note is None:
					yield paragraph
				else:
					note.append(paragraph)
				tokens.clear()
			
			if note is not None:
				assert note[0].commands[0].content[0] == "Note"
				title = " ".join(str(_t) for _t in note[0].commands[0].content)
				commands = []
				for subparagraph in note[1:]:
					if isinstance(subparagraph, Paragraph):
						commands.extend(subparagraph.commands)
					elif isinstance(subparagraph, Algorithm):
						commands.extend(subparagraph.instructions)
					elif isinstance(subparagraph, Grammar):
						commands.append(Cmd([subparagraph]))
					elif isinstance(subparagraph, Table):
						pass # TODO
					else:
						raise NotImplementedError(type(subparagraph).__name__)
				
				note = Note()
				note.title = title
				note.commands = commands
				if __debug__: note._validate()
				yield note
		
		elif xml.tag in [f'{{{self.html_namespace}}}ul', f'{{{self.html_namespace}}}ol']:
			paragraph = Paragraph()
			for li in xml:
				#tokens = list(self.tokenize_spec(li))
				tokens = []
				if li.text:
					tokens.extend(self.tokenize_string(li.text))
				for el in li:
					if el.tag in [f'{{{self.html_namespace}}}strong', f'{{{self.html_namespace}}}a', f'{{{self.html_namespace}}}em', f'{{{self.emu_namespace}}}not-ref']:
						if el.text:
							tokens.extend(self.tokenize_string(el.text))
					else:
						tokens.extend(self.token(el))
					if el.tail:
						tokens.extend(self.tokenize_string(el.tail))
				if tokens:
					paragraph.commands.append(Cmd(tokens))
			if paragraph.commands:
				yield paragraph
		
		elif xml.tag in [f'{{{self.emu_namespace}}}nt', f'{{{self.html_namespace}}}code', f'{{{self.emu_namespace}}}const', f'{{{self.emu_namespace}}}val', f'{{{self.html_namespace}}}var', f'{{{self.emu_namespace}}}eqn', f'{{{self.emu_namespace}}}grammar', f'{{{self.html_namespace}}}figure']:
			paragraph = Paragraph(self.commands(self.token(xml)))
			if __debug__: paragraph._validate()
			yield paragraph
		
		else:
			tokens = deque()
			
			if xml.text:
				tokens.extend(self.tokenize_string(xml.text))
			
			for child in xml:
				for subparagraph in self.paragraphs(child, level + 1):
					if isinstance(subparagraph, Paragraph):
						tokens.extend(chain.from_iterable(_command.content for _command in subparagraph.commands))
					else:
						tokens.append(subparagraph)
				if child.tail:
					tokens.extend(self.tokenize_string(child.tail))
			
			if tokens:
				paragraph = Paragraph(self.commands(tokens))
				if __debug__: paragraph._validate()
				yield paragraph
				tokens.clear()
	
	def commands(self, tokens):
		"Yield paragraph-level commands (not for algorithm)."
		fences = 0
		tchain = deque()
		for token in tokens:
			begin = False
			if not tchain or (tchain[-1] == '.' and fences == 0):
				if isinstance(token, str):
					if token[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in self.opening_fences:
						begin = True
				else:
					if isinstance(token, Reference) or isinstance(token, Aoid):
						begin = True
			
			if begin:
				if tchain:
					cmd = Cmd(tchain)
					if __debug__: cmd._validate()
					yield cmd
					tchain.clear()
			
			if isinstance(token, str) and token in self.opening_fences: fences += 1
			if isinstance(token, str) and token in self.closing_fences: fences -= 1
			tchain.append(token)
		
		if tchain:
			cmd = Cmd(tchain)
			if __debug__: cmd._validate()
			yield cmd
			tchain.clear()
	
	def algorithm(self, xml):
		if xml.tag == f'{{{self.emu_namespace}}}alg':
			if len(xml) != 1: raise ValueError("<emu:alg/> tag must have exactly 1 child that is <html:ol/> or <html:ul/>.")
			xml = xml[0]
		
		if xml.tag != f'{{{self.html_namespace}}}ol' and xml.tag != f'{{{self.html_namespace}}}ul':
			raise ValueError("The tag must be <emu:alg/>, <html:ol/> or <html:ul/>.")
		
		algorithm = Algorithm()
		
		for li in xml:
			if not isinstance(li.tag, str): continue
			if li.tag != f'{{{self.html_namespace}}}li':
				raise ValueError("List element must be <html:li/>.")
			
			tokens = list(self.tokenize_spec(li))
			if isinstance(tokens[-1], Algorithm | Table):
				tchain_mod = []
				for token in tokens[:-1]:
					tchain_mod.append(token)
				cmd = Cmd(tchain_mod, tokens[-1])
			else:
				tchain_mod = []
				for token in tokens:
					tchain_mod.append(token)
				cmd = Cmd(tchain_mod)
			algorithm.instructions.append(cmd)
		
		return algorithm
	
	def rule(self, xml):
		if xml.tag == f'{{{self.emu_namespace}}}t':
			if len(xml) == 0:
				rule = Terminal(self.extract_text(xml))
				yield rule
			else:
				raise NotImplementedError(xml_tounicode(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}nt':
			if not 0 <= len(xml) <= 2:
				raise NotImplementedError(xml_tounicode(xml))
			
			if len(xml):
				for n, child in enumerate(xml):
					if n == 0 and child.tag == f'{{{self.html_namespace}}}a':
						rule = Nonterminal(self.extract_text(child))
						yield rule
					elif n == 1 and child.tag == f'{{{self.emu_namespace}}}mods':
						for opt in child:
							if opt.tag == f'{{{self.emu_namespace}}}opt':
								yield Option('opt', ())
							elif opt.tag == f'{{{self.emu_namespace}}}params':
								assert opt.text[0] == '[' and opt.text[-1] == ']'
								text = opt.text[1:-1]
								yield Option('param', tuple(_parm.strip() for _parm in text.split(',')))
							else:
								raise NotImplementedError(xml_tounicode(opt))
					else:
						raise NotImplementedError(str(n) + " " + xml_tounicode(child))
			else:
				rule = Nonterminal(self.extract_text(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}constraints':
			pass
		
		elif xml.tag == f'{{{self.emu_namespace}}}gann':
			assert xml.text[0] == '[', repr(xml.text)
			text = xml.text[1:]
			if text.startswith('lookahead ∉'):
				yield Lookahead('reject', ()) # TODO
			elif text.startswith('lookahead ∈'):
				yield Lookahead('require', ()) # TODO
			elif text.startswith('lookahead ≠'):
				yield Lookahead('reject', ()) # TODO
			elif text.startswith('no'):
				yield Lookahead('reject', ()) # TODO
			elif text.startswith('empty'):
				yield Lookahead('empty', ())
			else:
				raise NotImplementedError(xml_tounicode(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}gmod':
			if xml.text.startswith('but not'):
				yield Option('but-not', ()) # TODO
			elif xml.text.startswith('but only if the MV of'):
				yield Option('MV-of', ()) # TODO
			else:
				raise NotImplementedError(xml_tounicode(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}gprose':
			if xml.text == "any Unicode code point with the Unicode property “ID_Start”":
				yield CharMatch('Start')
			elif xml.text == "any Unicode code point with the Unicode property “ID_Continue”":
				yield CharMatch('Continue')
			elif xml.text == "any Unicode code point":
				yield CharMatch('Any')
			elif xml.text == "any Unicode code point in the inclusive interval from U+D800 to U+DBFF":
				yield CharMatch('D800-DBFF')
			elif xml.text == "any Unicode code point in the inclusive interval from U+DC00 to U+DFFF":
				yield CharMatch('DC00-DFFF')
			elif xml.text[0] == "<" and xml.text[-1] == ">":
				yield CharMatch(xml.text[1:-1])
			else:
				raise NotImplementedError(xml_tounicode(xml))
		
		else:
			raise NotImplementedError(xml_tounicode(xml))
	
	def production(self, head, rules):
		o = set()
		result = []
		for n, rule in enumerate(rules):
			if isinstance(rule, Terminal) or isinstance(rule, Nonterminal) or isinstance(rule, Lookahead) or isinstance(rule, CharMatch):
				if n:
					result.append(frozenset(o))
					o = set()
				result.append(rule)
			elif isinstance(rule, Option):
				o.add(rule)
			else:
				raise NotImplementedError(repr(rule))
		else:
			result.append(frozenset(o))
		
		symbols = [result[_n] for _n in range(0, len(result), 2)]
		options = [result[_n] for _n in range(1, len(result), 2)]
		
		if not options or (len(set(options)) == 1 and options[0] == frozenset()):
			return Production(head, tuple(symbols), None)
		else:
			return Production(head, tuple(symbols), tuple(options))
	
	def grammar(self, xml):
		if xml.tag != f'{{{self.emu_namespace}}}grammar':
			raise ValueError("The tag must be <emu:grammar/>.")
		
		type_ = [_prod_child for _prod_child in xml if _prod_child.tag == f'{{{self.emu_namespace}}}production'][0].attrib.get('type', '')
		
		productions = []
		
		for prod_child in xml:
			if prod_child.tag != f'{{{self.emu_namespace}}}production':
				raise NotImplementedError			
			
			one_of = 'oneof' in prod_child.attrib
			
			if not one_of:
				assert len(prod_child) >= 3 and prod_child[0].tag == f'{{{self.emu_namespace}}}nt' and prod_child[1].tag == f'{{{self.emu_namespace}}}geq' and all(_subchild.tag == f'{{{self.emu_namespace}}}rhs' for _subchild in prod_child[2:]), xml_tounicode(xml)
			else:
				assert len(prod_child) == 4 and prod_child[0].tag == f'{{{self.emu_namespace}}}nt' and prod_child[1].tag == f'{{{self.emu_namespace}}}geq' and prod_child[2].tag == f'{{{self.emu_namespace}}}oneof' and prod_child[3].tag == f'{{{self.emu_namespace}}}rhs', xml_tounicode(xml)
			
			assert type_ == prod_child.attrib.get('type', '')
			
			head = Nonterminal(self.extract_text(prod_child[0][0]))
			
			if one_of:
				for child in prod_child[3]:
					rules = list(self.rule(child))
					production = self.production(head, rules)
					productions.append(production)
			
			else:
				for rhs in prod_child[2:]:
					if rhs.tag !=  f'{{{self.emu_namespace}}}rhs':
						raise NotImplementedError(xml_tounicode(rhs))
					
					rules = list(chain.from_iterable(self.rule(_child) for _child in rhs))
					production = self.production(head, rules)
					productions.append(production)
		
		return Grammar(type_, tuple(productions))
	
	def table(self, xml):
		if xml.tag != f'{{{self.emu_namespace}}}table':
			raise ValueError
		
		#if 'id' not in xml.attrib:
		#	print(xml_tounicode(xml))
		
		table = Table(xml.attrib['id'] if 'id' in xml.attrib else '')
		
		#print(xml.find.__doc__)
		htable = xml.find('.//html:table', namespaces={'html':self.html_namespace, 'emu':self.emu_namespace})
		
		try:
			thead = [_child for _child in htable if _child.tag == f'{{{self.html_namespace}}}thead'][0]
		except IndexError:
			pass
		else:
			for tr in thead:
				if tr.tag != f'{{{self.html_namespace}}}tr': continue
				row = []
				for th in tr:
					if th.tag != f'{{{self.html_namespace}}}th': continue
					pars = list(self.paragraphs(th))
					if len(pars) == 1:
						row.append(pars[0])
					else:
						paragraph = Paragraph(tuple(chain.from_iterable(_p.commands for _p in pars)))
						row.append(paragraph)
				table.head.append(row)
		
		try:
			tbody = [_child for _child in htable if _child.tag == f'{{{self.html_namespace}}}tbody'][0]
		except IndexError:
			tbody = htable
		
		for tr in tbody:
			if tr.tag != f'{{{self.html_namespace}}}tr': continue
			row = []
			for td in tr:
				if td.tag != f'{{{self.html_namespace}}}td': continue
				pars = list(self.paragraphs(td))
				if len(pars) == 1:
					row.append(pars[0])
				else:
					paragraph = Paragraph(tuple(chain.from_iterable(_p.commands for _p in pars)))
					row.append(paragraph)
			table.body.append(row)
		
		if __debug__: table._validate()
		return table
	
	whitespace = ' \t\r\n\xa0'
	separators = '.,;%:+-—/~"'
	opening_fences = '([{«“'
	closing_fences = ')]}»”'
	punctuation = whitespace + opening_fences + closing_fences + separators
	
	def token(self, xml):
		if not isinstance(xml.tag, str):
			raise NotImplementedError
		
		if xml.tag == f'{{{self.emu_namespace}}}nt':
			yield Nonterminal(self.extract_text(xml))
		
		elif xml.tag == f'{{{self.html_namespace}}}code':
			yield Terminal(self.extract_text(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}const':
			yield Enum_(self.extract_text(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}val':
			yield Constant(self.extract_text(xml))
		
		elif xml.tag == f'{{{self.html_namespace}}}var':
			text = self.extract_text(xml).strip()
			if text.startswith('[[') and text.endswith(']]'):
				yield ObjectField(text[2:-2])
			else:
				yield Variable(text)
		
		elif xml.tag == f'{{{self.emu_namespace}}}eqn':
			yield Equation(self.extract_text(xml))
		
		elif xml.tag == f'{{{self.emu_namespace}}}grammar':
			yield self.grammar(xml)
		
		elif xml.tag == f'{{{self.html_namespace}}}figure':
			pass # TODO
			#elements = [_subxml for _subxml in xml if isinstance(_subxml.tag, str)]
			#if len(elements) != 1:
			#	raise ValueError(f"Expected exactly 1 element as a child of <html:figure/> tag. Got: {xml_tounicode(xml)}")
			#if elements[0].tag != f'{{{self.html_namespace}}}table':
			#	raise ValueError(xml_tounicode(xml))
			#
			#yield self.table(elements[0])
		
		elif xml.tag == f'{{{self.emu_namespace}}}xref':
			if 'aoid' in xml.attrib:
				yield Aoid(xml.attrib['aoid'], tuple(self.tokenize_string(self.extract_text(xml))))
			elif 'href' in xml.attrib:
				yield Reference(xml.attrib['href'], tuple(self.tokenize_string(self.extract_text(xml))))
			else:
				raise NotImplementedError("XREF doesn't have href nor aoid.")
		
		elif xml.tag == f'{{{self.html_namespace}}}dfn':
			yield Definition(xml.attrib.get('id', ''), tuple(self.tokenize_string(xml.text)) if xml.text else None)
		
		elif xml.tag == f'{{{self.html_namespace}}}sup':
			yield SpecialSymbol.Sup
			yield SpecialSymbol.Bra
			yield from self.tokenize_spec(xml)
			yield SpecialSymbol.Ket
		
		elif xml.tag == f'{{{self.html_namespace}}}sub':
			yield SpecialSymbol.Sub
			yield SpecialSymbol.Bra
			yield from self.tokenize_spec(xml)
			yield SpecialSymbol.Ket
		
		else:
			raise NotImplementedError(xml_tounicode(xml))
	
	def tokenize_string(self, text):
		pieces = [text.strip()]
		
		for ch in self.punctuation:
			pieces = self.split_over_character(ch, pieces)
		
		for token in pieces:
			if isinstance(token, str):
				if token.isdigit():
					token = Digits(token)
				elif token.startswith("0x") and all(_ch in "0123456789abcdefABCDEF" for _ch in token[2:]):
					token = HexDigits(token)
				elif len(token) == 4 and  all(_ch in "0123456789ABCDEF" for _ch in token):
					token = HexDigits(token)
			
			yield token
	
	def tokenize_spec(self, xml):
		if xml.tag in self.subparagraph_tags and xml.tag != f'{{{self.html_namespace}}}span':
			raise ValueError(f"Subparagraph tags except <html:span/> are not subject to tokenization. {xml.tag}")
		
		if xml.text is not None:
			yield from self.tokenize_string(xml.text)
		
		for child in xml:
			if not isinstance(child.tag, str):
				pass
			
			elif child.tag in [f'{{{self.html_namespace}}}ul', f'{{{self.html_namespace}}}ol']:
				yield self.algorithm(child)
			
			elif child.tag in [
					f'{{{self.html_namespace}}}span', f'{{{self.html_namespace}}}em', f'{{{self.html_namespace}}}b',
					f'{{{self.emu_namespace}}}not-ref', f'{{{self.html_namespace}}}a', f'{{{self.html_namespace}}}i', f'{{{self.html_namespace}}}strong'
				]:
				yield from self.tokenize_spec(child)
			
			elif child.tag == f'{{{self.html_namespace}}}br':
				pass
			
			else:
				yield from self.token(child)
			
			if child.tail is not None:
				yield from self.tokenize_string(child.tail)
	
	@classmethod
	def split_over_character(cls, ch, strs):
		r = []
		for s in strs:
			s = s.strip()
			p = s.split(ch)
			r.extend(_p.strip() for _p in sum(zip(p, [ch] * len(p)), ())[:-1] if _p)
		return [_r for _r in r if _r]
	
	@classmethod
	def extract_text(cls, node):
		t = []
		if node.text:
			t.append(node.text)
		for child in node:
			if isinstance(child.tag, str):
				t.append(cls.extract_text(child))
			if child.tail:
				t.append(child.tail)
		return "".join(t)
	
	@classmethod
	def extract_tags(cls, node, tag):
		if node.tag == tag:
			yield node
		else:
			for child in node:
				yield from cls.extract_tags(child, tag)


if __name__ == '__main__':
	from pathlib import Path
	from pickle import dump
	
	specification = Specification()
	for specfile in sorted(Path('specification').iterdir()):
		print(specfile)
		specification.parse(xml_frombytes(specfile.read_bytes()))
	
	assert specification.find_ref('#sec-completion-record-specification-type').title == "The Completion Record Specification Type"
	assert specification.find_clause_with_chapter((6, 1, 2)).title == "The Null Type"
	
	#for clause in specification.find_clauses_with_title("Runtime Semantics: Evaluation"):
	#	for level, line in clause._print():
	#		print(" " * level + line)
	#	print("")
	
	pickle_dir = Path('pickle')
	pickle_dir.mkdir(exist_ok=True)
	
	with (pickle_dir / 'specification.pickle').open('wb') as fd:
		dump(specification, fd)
	
