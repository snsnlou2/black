

class MyClass():
    ' Multiline\n  class docstring\n  '

    def method(self):
        'Multiline\n    method docstring\n    '
        pass

def foo():
    'This is a docstring with             \n  some lines of text here\n  '
    return

def bar():
    'This is another docstring\n  with more lines of text\n  '
    return

def baz():
    '"This" is a string with some\n  embedded "quotes"'
    return

def troz():
    'Indentation with tabs\n\tis just as OK\n\t'
    return

def zort():
    'Another\n        multiline\n        docstring\n        '
    pass

def poit():
    '\n  Lorem ipsum dolor sit amet.       \n\n  Consectetur adipiscing elit:\n   - sed do eiusmod tempor incididunt ut labore\n   - dolore magna aliqua\n     - enim ad minim veniam\n     - quis nostrud exercitation ullamco laboris nisi\n   - aliquip ex ea commodo consequat\n  '
    pass

def under_indent():
    '\n  These lines are indented in a way that does not\nmake sense.\n  '
    pass

def over_indent():
    '\n  This has a shallow indent\n    - But some lines are deeper\n    - And the closing quote is too deep\n    '
    pass

def single_line():
    'But with a newline after it!\n\n    '
    pass

def this():
    "\n    'hey ho'\n    "

def that():
    ' "hey yah" '

def and_that():
    '\n  "hey yah" '

def and_this():
    ' \n  "hey yah"'

def believe_it_or_not_this_is_in_the_py_stdlib():
    ' \n"hey yah"'

def ignored_docstring():
    'a => b'

def docstring_with_inline_tabs_and_space_indentation():
    'hey\n\n    tab\tseparated\tvalue\n    \ttab at start of line and then a tab\tseparated\tvalue\n    \t\t\t\tmultiple tabs at the beginning\tand\tinline\n    \t \t  \tmixed tabs and spaces at beginning. next line has mixed tabs and spaces only.\n    \t\t\t \t  \t\t\n    line ends with some tabs\t\t\n    '

def docstring_with_inline_tabs_and_tab_indentation():
    'hey\n\n\ttab\tseparated\tvalue\n\t\ttab at start of line and then a tab\tseparated\tvalue\n\t\t\t\t\tmultiple tabs at the beginning\tand\tinline\n\t\t \t  \tmixed tabs and spaces at beginning. next line has mixed tabs and spaces only.\n\t\t\t\t \t  \t\t\n\tline ends with some tabs\t\t\n\t'
    pass

class MyClass():
    'Multiline\n    class docstring\n    '

    def method(self):
        'Multiline\n        method docstring\n        '
        pass

def foo():
    'This is a docstring with\n    some lines of text here\n    '
    return

def bar():
    'This is another docstring\n    with more lines of text\n    '
    return

def baz():
    '"This" is a string with some\n    embedded "quotes"'
    return

def troz():
    'Indentation with tabs\n    is just as OK\n    '
    return

def zort():
    'Another\n    multiline\n    docstring\n    '
    pass

def poit():
    '\n    Lorem ipsum dolor sit amet.\n\n    Consectetur adipiscing elit:\n     - sed do eiusmod tempor incididunt ut labore\n     - dolore magna aliqua\n       - enim ad minim veniam\n       - quis nostrud exercitation ullamco laboris nisi\n     - aliquip ex ea commodo consequat\n    '
    pass

def under_indent():
    '\n      These lines are indented in a way that does not\n    make sense.\n    '
    pass

def over_indent():
    '\n    This has a shallow indent\n      - But some lines are deeper\n      - And the closing quote is too deep\n    '
    pass

def single_line():
    'But with a newline after it!'
    pass

def this():
    "\n    'hey ho'\n    "

def that():
    ' "hey yah" '

def and_that():
    '\n    "hey yah" '

def and_this():
    '\n    "hey yah"'

def believe_it_or_not_this_is_in_the_py_stdlib():
    '\n    "hey yah"'

def ignored_docstring():
    'a => b'

def docstring_with_inline_tabs_and_space_indentation():
    'hey\n\n    tab\tseparated\tvalue\n        tab at start of line and then a tab\tseparated\tvalue\n                                multiple tabs at the beginning\tand\tinline\n                        mixed tabs and spaces at beginning. next line has mixed tabs and spaces only.\n\n    line ends with some tabs\n    '

def docstring_with_inline_tabs_and_tab_indentation():
    'hey\n\n    tab\tseparated\tvalue\n            tab at start of line and then a tab\tseparated\tvalue\n                                    multiple tabs at the beginning\tand\tinline\n                            mixed tabs and spaces at beginning. next line has mixed tabs and spaces only.\n\n    line ends with some tabs\n    '
    pass
