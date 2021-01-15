
while True:
    if something.changed:
        do.stuff()
for i in range(100):
    if ((i % 33) == 0):
        break
    print(i)
with open(some_temp_file) as f:
    data = f.read()
try:
    with open(some_other_file) as w:
        w.write(data)
except OSError:
    print('problems')
import sys

def wat():
    ...

@deco1
@deco2(with_args=True)
@deco3
def decorated1():
    ...

@deco1
@deco2(with_args=True)
def decorated1():
    ...
some_instruction

def g():
    ...
if (__name__ == '__main__'):
    main()
