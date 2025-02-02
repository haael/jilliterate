#!/usr/bin/python3


__all__ = 'Token', 'Digits', 'HexDigits', 'Enum_', 'Constant', 'Variable', 'ObjectField', 'Reference', 'Aoid', 'Definition', 'Equation', 'SpecialSymbol', \
          'Terminal', 'Nonterminal', 'Lookahead', 'CharMatch', 'Option', 'Production', 'Grammar'


from enum import Enum
#from typing import Self
from dataclasses import dataclass

from datatypes import *


class Token:
	def render(self, fn, path=(), index=()):
		yield from fn(self, path + (self.__class__.__name__,), index)


@dataclass(frozen=True)
class Digits(Token):
	value: str


@dataclass(frozen=True)
class HexDigits(Token):
	value: str


@dataclass(frozen=True)
class Terminal(Token):
	value: str


@dataclass(frozen=True)
class Nonterminal(Token):
	value: str


@dataclass(frozen=True)
class Enum_(Token):
	value: str


@dataclass(frozen=True)
class Constant(Token):
	value: str


@dataclass(frozen=True)
class Variable(Token):
	value: str


@dataclass(frozen=True)
class ObjectField(Token):
	value: str


@dataclass(frozen=True)
class Reference(Token):
	href: str
	content: tuple
	
	def render(self, fn, path=(), index=()):
		path = path + ('Reference',)
		if self.content is not None:
			content = (token if isinstance(token, str) else token.render(fn, path, index + (n,)) for (n, token) in enumerate(self.content))
		else:
			content = None
		yield from fn(self, path, index, content=content)


@dataclass(frozen=True)
class Aoid(Token):
	aoid: str
	content: tuple
	
	def render(self, fn, path=(), index=()):
		path = path + ('Aoid',)
		if self.content is not None:
			content = (token if isinstance(token, str) else token.render(fn, path, index + (n,)) for (n, token) in enumerate(self.content))
		else:
			content = None
		yield from fn(self, path, index, content=content)


@dataclass(frozen=True)
class Definition(Token):
	id_: str
	content: tuple
	
	def render(self, fn, path=(), index=()):
		path = path + ('Definition',)
		if self.content is not None:
			content = (token if isinstance(token, str) else token.render(fn, path, index + (n,)) for (n, token) in enumerate(self.content))
		else:
			content = None
		yield from fn(self, path, index, content=content)


@dataclass(frozen=True)
class Equation(Token):
	content: str
	
	def render(self, fn, path=(), index=()):
		path = path + ('Equation',)
		if self.content is not None:
			content = (token if isinstance(token, str) else token.render(fn, path, index + (n,)) for (n, token) in enumerate(self.content))
		else:
			content = None
		yield from fn(self, path, index, content=content)


SpecialSymbol = Enum('SpecialSymbol', 'Sup Sub Bra Ket')


@dataclass(frozen=True)
class Lookahead(Token):
	type_: str
	productions: tuple
	
	def render(self, fn, path=(), index=()):
		path = path + ('Lookahead',)
		if self.productions is not None:
			productions = (token if isinstance(token, str) else token.render(fn, path, index + (n,)) for (n, token) in enumerate(self.productions))
		else:
			productions = None
		yield from fn(self, path, index, productions=productions)


@dataclass(frozen=True)
class CharMatch(Token):
	value: str


@dataclass(frozen=True)
class Option:
	type_: str
	arg: tuple[str]
	
	def render(self, fn, path=(), index=()):
		path = path + ('Option',)
		yield from fn(self, path, index)


@dataclass(frozen=True)
class Production:
	head: Nonterminal
	symbols: tuple[Terminal | Nonterminal]
	options: tuple[frozenset[Option]] | None
	
	def __init__(self, head, symbols, options=None):
		if not isinstance(symbols, tuple):
			symbols = tuple(symbols)
		
		if options is None:
			pass
		elif not options or (len(set(options)) == 1 and options[0] == frozenset()):
			options = None
		elif not isinstance(options, tuple):
			options = tuple(options)
		
		object.__setattr__(self, 'head', head)
		object.__setattr__(self, 'symbols', symbols)
		object.__setattr__(self, 'options', options)
	
	def __repr__(self):
		if self.options is not None:
			return self.__class__.__name__ + '(head=' + repr(self.head) + ', symbols=[' + ', '.join(repr(_s) for _s in self.symbols) + '], options=[' + ', '.join(repr(_s) for _s in self.options) + '])'
		else:
			return self.__class__.__name__ + '(head=' + repr(self.head) + ', symbols=[' + ', '.join(repr(_s) for _s in self.symbols) + '])'
	
	def _print(self, level=0):
		yield level, repr(self)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Production',)
		head = self.head.render(fn, path, index)
		symbols = (symbol.render(fn, path, index + (n,)) for (n, symbol) in enumerate(self.symbols))
		if self.options is not None:
			options = ((_option.render(fn, path, index + (n,)) for _option in _options) for (n, _options) in enumerate(self.options))
		else:
			options = None
		yield from fn(self, path, index, head=head, symbols=symbols, options=options)


@dataclass(frozen=True)
class Grammar:
	type_: str
	productions: tuple[Production]
	
	def _print(self, level=0):
		yield level, "* " + (((self.type_ + " ") if self.type_ else "") + "grammar:").capitalize()
		for production in self.productions:
			yield from production._print(level + 1)
	
	def render(self, fn, path=(), index=()):
		path = path + ('Grammar',)
		productions = (production.render(fn, path, index + (n,)) for (n, production) in enumerate(self.productions))
		yield from fn(self, path, index, productions=productions)


