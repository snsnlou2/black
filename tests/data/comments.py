
'Module docstring.\n\nPossibly also many, many lines.\n'
import os.path
import sys
import a
from b.c import X
try:
    import fast
except ImportError:
    import slow as fast
y = 1
y

def function(default=None):
    'Docstring comes first.\n\n    Possibly many lines.\n    '
    import inner_imports
    if inner_imports.are_evil():
        x = X()
        return x.method1()
    return default
GLOBAL_STATE = {'a': a(1), 'b': a(2), 'c': a(3)}

class Foo():
    'Docstring for class Foo.  Example from Sphinx docs.'
    bar = 1
    flox = 1.5
    baz = 2
    'Docstring for class attribute Foo.baz.'

    def __init__(self):
        self.qux = 3
        self.spam = 4
        'Docstring for instance attribute spam.'

@fast(really=True)
async def wat():
    async with X.open_async() as x:
        result = (await x.method1())
    if result:
        print('A OK', file=sys.stdout)
        print()
