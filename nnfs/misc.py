# Miscellaneous functions

from inspect import stack

def logg(*args, extra_newlines=True):
    if extra_newlines:
        print()

    # stack behavior: main.py -> intermediate_a.py -> ... -> intermediate_x.py (contains logg() call) -> logg.py
    # stack[1] will contain data about intermediate_x.py
    frameinfo = stack()[1]
    print('***Log msg at {}, line {}***'.format(frameinfo.filename, frameinfo.lineno))
    print(*args)
    
    if extra_newlines:
        print()
