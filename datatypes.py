#!/usr/bin/python3

"Some basic types used by other modules."


__all__ = 'TypeChecked', 'collection_of', 'tuple_of', 'list_of', 'set_of', 'TypedRecord', 'GatheringProperty', 'FilteringProperty', 'subsets', 'unique'


from typing import Self
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Iterable


class TypeChecked(type):
	def __instancecheck__(self, value):
		if not any(isinstance(value, _base) for _base in self.__bases__):
			return False
		
		if self.__element_type != None:
			if not all(isinstance(_item, self.__element_type) for _item in value):
				return False
		
		for attr, required_type in self.__annotations__.items():
			try:
				if not isinstance(getattr(value, attr), required_type):
					return False
			except AttributeError:
				return False
		
		return True


def collection_of(T):
	class CollectionOfType(metaclass=TypeChecked):
		_TypeChecked__element_type = T
	return CollectionOfType


def tuple_of(T):
	class TupleOfType(tuple, metaclass=TypeChecked):
		_TypeChecked__element_type = T
	return TupleOfType


def list_of(T):
	class ListOfType(list, metaclass=TypeChecked):
		_TypeChecked__element_type = T
	return ListOfType


def set_of(T):
	class SetOfType(set, metaclass=TypeChecked):
		_TypeChecked__element_type = T
	return SetOfType


class TypedRecord(metaclass=TypeChecked):
	_TypeChecked__element_type = None
	
	def __setattr__(self, attr, value):
		try:
			required_type = self.__annotations__[attr]
		except KeyError:
			raise TypeError(f"No such field in record {self.__class__.__name__}: `{attr}`.")
		else:
			if not isinstance(value, required_type):
				if hasattr(required_type, '_TypeChecked__element_type'):
					if isinstance(value, Iterable):
						raise TypeError(f"Invalid type for field `{attr}` of class {self.__class__.__name__}. Required: {required_type.__name__}({required_type._TypeChecked__element_type}), got: {type(value)}{[type(_item).__name__ for _item in value]}.")
					else:
						raise TypeError(f"Invalid type for field `{attr}` of class {self.__class__.__name__}. Required: {required_type.__name__}({required_type._TypeChecked__element_type}), got: {type(value)}.")
				else:
					raise TypeError(f"Invalid type for field `{attr}` of class {self.__class__.__name__}. Required: {required_type.__name__}, got: {type(value).__name__}.")
		super().__setattr__(attr, value)
	
	def _validate(self):
		for attr, required_type in self.__annotations__.items():
			if type(required_type) == TypeChecked and required_type._TypeChecked__element_type == Self:
				required_type = collection_of(self.__class__)
			try:
				if not isinstance(getattr(self, attr), required_type):
					raise TypeError(f"Object {repr(self)} of type {self.__class__.__name__} did not pass validation.")
			except AttributeError:
				raise TypeError(f"Missing field `{attr}` in object of type {self.__class__.__name__}.")


def unique(old_comethod):
	def new_comethod(*args, **kwargs):
		seen = set()
		for item in old_comethod(*args, *kwargs):
			if item not in seen:
				yield item
				seen.add(item)
	return new_comethod


class GatheringProperty:
	def __init__(self, field, children):
		self.field = field
		self.children = children
	
	@unique
	def __get__(self, instance, owner=None):
		for attr in self.children:
			for item in getattr(instance, attr):
				yield from getattr(item, self.field)


class FilteringProperty:
	def __init__(self, field, type_):
		self.field = field
		self.type_ = type_
	
	@unique
	def __get__(self, instance, owner=None):
		for item in getattr(instance, self.field):
			if isinstance(item, self.type_):
				yield item


def subsets(l):
	l = list(l)
	for n in range(len(l) + 1):
		yield from combinations(l, n)

