
'Export the Python grammar and symbols.'
import os
from typing import Union
from .pgen2 import token
from .pgen2 import driver
from .pgen2.grammar import Grammar

class Symbols(object):

    def __init__(self, grammar):
        "Initializer.\n\n        Creates an attribute for each grammar symbol (nonterminal),\n        whose value is the symbol's type (an int >= 256).\n        "
        for (name, symbol) in grammar.symbol2number.items():
            setattr(self, name, symbol)

class _python_symbols(Symbols):

class _pattern_symbols(Symbols):

def initialize(cache_dir=None):
    global python_grammar
    global python_grammar_no_print_statement
    global python_grammar_no_print_statement_no_exec_statement
    global python_grammar_no_print_statement_no_exec_statement_async_keywords
    global python_symbols
    global pattern_grammar
    global pattern_symbols
    _GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), 'Grammar.txt')
    _PATTERN_GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), 'PatternGrammar.txt')
    python_grammar = driver.load_packaged_grammar('blib2to3', _GRAMMAR_FILE, cache_dir)
    python_symbols = _python_symbols(python_grammar)
    python_grammar_no_print_statement = python_grammar.copy()
    del python_grammar_no_print_statement.keywords['print']
    python_grammar_no_print_statement_no_exec_statement = python_grammar.copy()
    del python_grammar_no_print_statement_no_exec_statement.keywords['print']
    del python_grammar_no_print_statement_no_exec_statement.keywords['exec']
    python_grammar_no_print_statement_no_exec_statement_async_keywords = python_grammar_no_print_statement_no_exec_statement.copy()
    python_grammar_no_print_statement_no_exec_statement_async_keywords.async_keywords = True
    pattern_grammar = driver.load_packaged_grammar('blib2to3', _PATTERN_GRAMMAR_FILE, cache_dir)
    pattern_symbols = _pattern_symbols(pattern_grammar)
