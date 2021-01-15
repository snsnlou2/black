
import asyncio
import sys
from third_party import X, Y, Z
from library import some_connection, some_decorator
from third_party import X, Y, Z
f'trigger 3.6 mode'

def func_no_args():
    a
    b
    c
    if True:
        raise RuntimeError
    if False:
        ...
    for i in range(10):
        print(i)
        continue
    exec('new-style exec', {}, {})
    return None

async def coroutine(arg, exec=False):
    'Single-line docstring. Multiline is harder to reformat.'
    async with some_connection() as conn:
        (await conn.do_what_i_mean('SELECT bobby, tables FROM xkcd', timeout=2))
    (await asyncio.sleep(1))

@asyncio.coroutine
@some_decorator(with_args=True, many_args=[1, 2, 3])
def function_signature_stress_test(number, no_annotation=None, text='default', *, debug=False, **kwargs):
    return text[number:(- 1)]

def spaces(a=1, b=(), c=[], d={}, e=True, f=(- 1), g=(1 if False else 2), h='', i=''):
    offset = attr.ib(default=attr.Factory((lambda : _r.uniform(1, 2))))
    assert (task._cancel_stack[:len(old_stack)] == old_stack)

def spaces_types(a=1, b=(), c=[], d={}, e=True, f=(- 1), g=(1 if False else 2), h='', i=''):
    ...

def spaces2(result=_core.Value(None)):
    ...
something = {key: 'value'}

def subscriptlist():
    atom[('some big and', 'complex subscript', (goes + here), andhere)]

def import_as_names():
    from hello import a, b
    'unformatted'

def testlist_star_expr():
    (a, b) = *hello
    'unformatted'

def yield_expr():
    (yield hello)
    'unformatted'
    'formatted'
    (yield hello)
    'unformatted'

def example(session):
    result = session.query(models.Customer.id).filter((models.Customer.account_id == account_id), (models.Customer.email == email_address)).order_by(models.Customer.id.asc()).all()

def off_and_on_without_data():
    'All comments here are technically on the same prefix.\n\n    The comments between will be formatted. This is a known limitation.\n    '
    pass

def on_and_off_broken():
    'Another known limitation.'
    this = should.not_be.formatted()
    and_ = (indeed.it is not formatted)
    because.the.handling.inside.generate_ignored_nodes()
    now.considers.multiple.fmt.directives.within.one.prefix

def long_lines():
    if True:
        typedargslist.extend(gen_annotated_params(ast_args.kwonlyargs, ast_args.kw_defaults, parameters, implicit_default=True))
        a = unnecessary_bracket()
    _type_comment_re = re.compile('\n        ^\n        [\\t ]*\n        \\#[ ]type:[ ]*\n        (?P<type>\n            [^#\\t\\n]+?\n        )\n        (?<!ignore)     # note: this will force the non-greedy + in <type> to match\n                        # a trailing space which is why we need the silliness below\n        (?<!ignore[ ]{1})(?<!ignore[ ]{2})(?<!ignore[ ]{3})(?<!ignore[ ]{4})\n        (?<!ignore[ ]{5})(?<!ignore[ ]{6})(?<!ignore[ ]{7})(?<!ignore[ ]{8})\n        (?<!ignore[ ]{9})(?<!ignore[ ]{10})\n        [\\t ]*\n        (?P<nl>\n            (?:\\#[^\\n]*)?\n            \\n?\n        )\n        $\n        ', (re.MULTILINE | re.VERBOSE))

def single_literal_yapf_disable():
    'Black does not support this.'
    BAZ = {(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)}
cfg.rule('Default', 'address', xxxx_xxxx=['xxx-xxxxxx-xxxxxxxxxx'], xxxxxx='xx_xxxxx', xxxxxxx='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', xxxxxxxxx_xxxx=True, xxxxxxxx_xxxxxxxxxx=False, xxxxxx_xxxxxx=2, xxxxxx_xxxxx_xxxxxxxx=70, xxxxxx_xxxxxx_xxxxx=True, xxxxxxx_xxxxxxxxxxxx={'xxxxxxxx': {'xxxxxx': False, 'xxxxxxx': False, 'xxxx_xxxxxx': 'xxxxx'}, 'xxxxxxxx-xxxxx': {'xxxxxx': False, 'xxxxxxx': True, 'xxxx_xxxxxx': 'xxxxxx'}}, xxxxxxxxxx_xxxxxxxxxxx_xxxxxxx_xxxxxxxxx=5)
(yield 'hello')
l = [1, 2, 3]
d = {'a': 1, 'b': 2}
import asyncio
import sys
from third_party import X, Y, Z
from library import some_connection, some_decorator
from third_party import X, Y, Z
f'trigger 3.6 mode'

def func_no_args():
    a
    b
    c
    if True:
        raise RuntimeError
    if False:
        ...
    for i in range(10):
        print(i)
        continue
    exec('new-style exec', {}, {})
    return None

async def coroutine(arg, exec=False):
    'Single-line docstring. Multiline is harder to reformat.'
    async with some_connection() as conn:
        (await conn.do_what_i_mean('SELECT bobby, tables FROM xkcd', timeout=2))
    (await asyncio.sleep(1))

@asyncio.coroutine
@some_decorator(with_args=True, many_args=[1, 2, 3])
def function_signature_stress_test(number, no_annotation=None, text='default', *, debug=False, **kwargs):
    return text[number:(- 1)]

def spaces(a=1, b=(), c=[], d={}, e=True, f=(- 1), g=(1 if False else 2), h='', i=''):
    offset = attr.ib(default=attr.Factory((lambda : _r.uniform(1, 2))))
    assert (task._cancel_stack[:len(old_stack)] == old_stack)

def spaces_types(a=1, b=(), c=[], d={}, e=True, f=(- 1), g=(1 if False else 2), h='', i=''):
    ...

def spaces2(result=_core.Value(None)):
    ...
something = {key: 'value'}

def subscriptlist():
    atom[('some big and', 'complex subscript', (goes + here), andhere)]

def import_as_names():
    from hello import a, b
    'unformatted'

def testlist_star_expr():
    (a, b) = *hello
    'unformatted'

def yield_expr():
    (yield hello)
    'unformatted'
    'formatted'
    (yield hello)
    'unformatted'

def example(session):
    result = session.query(models.Customer.id).filter((models.Customer.account_id == account_id), (models.Customer.email == email_address)).order_by(models.Customer.id.asc()).all()

def off_and_on_without_data():
    'All comments here are technically on the same prefix.\n\n    The comments between will be formatted. This is a known limitation.\n    '
    pass

def on_and_off_broken():
    'Another known limitation.'
    this = should.not_be.formatted()
    and_ = (indeed.it is not formatted)
    because.the.handling.inside.generate_ignored_nodes()
    now.considers.multiple.fmt.directives.within.one.prefix

def long_lines():
    if True:
        typedargslist.extend(gen_annotated_params(ast_args.kwonlyargs, ast_args.kw_defaults, parameters, implicit_default=True))
        a = unnecessary_bracket()
    _type_comment_re = re.compile('\n        ^\n        [\\t ]*\n        \\#[ ]type:[ ]*\n        (?P<type>\n            [^#\\t\\n]+?\n        )\n        (?<!ignore)     # note: this will force the non-greedy + in <type> to match\n                        # a trailing space which is why we need the silliness below\n        (?<!ignore[ ]{1})(?<!ignore[ ]{2})(?<!ignore[ ]{3})(?<!ignore[ ]{4})\n        (?<!ignore[ ]{5})(?<!ignore[ ]{6})(?<!ignore[ ]{7})(?<!ignore[ ]{8})\n        (?<!ignore[ ]{9})(?<!ignore[ ]{10})\n        [\\t ]*\n        (?P<nl>\n            (?:\\#[^\\n]*)?\n            \\n?\n        )\n        $\n        ', (re.MULTILINE | re.VERBOSE))

def single_literal_yapf_disable():
    'Black does not support this.'
    BAZ = {(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)}
cfg.rule('Default', 'address', xxxx_xxxx=['xxx-xxxxxx-xxxxxxxxxx'], xxxxxx='xx_xxxxx', xxxxxxx='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', xxxxxxxxx_xxxx=True, xxxxxxxx_xxxxxxxxxx=False, xxxxxx_xxxxxx=2, xxxxxx_xxxxx_xxxxxxxx=70, xxxxxx_xxxxxx_xxxxx=True, xxxxxxx_xxxxxxxxxxxx={'xxxxxxxx': {'xxxxxx': False, 'xxxxxxx': False, 'xxxx_xxxxxx': 'xxxxx'}, 'xxxxxxxx-xxxxx': {'xxxxxx': False, 'xxxxxxx': True, 'xxxx_xxxxxx': 'xxxxxx'}}, xxxxxxxxxx_xxxxxxxxxxx_xxxxxxx_xxxxxxxxx=5)
(yield 'hello')
l = [1, 2, 3]
d = {'a': 1, 'b': 2}
