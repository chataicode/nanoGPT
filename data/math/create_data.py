import re
import random
from tqdm import tqdm
import os


def inference_core(expr):
    expr = expr[1:-1]
    F = expr.split(' ')[0]
    V = expr.split(' ')[1:]
    if len(V) == 1:
        a = V[0]
        if F == 'f':
            return a[::-1]
        elif F == '-':
            if a[0] != '-':
                return f'-{a}'
            else:
                return a[1:]
        elif F in '-+*':
            return a
        elif F[0] == 'e':
            if '-' == a[0]:
                sign = '-'
                a = a[1:]
            else:
                sign = ''
            if '.' not in a:
                a += '.'
            a1, a2 = a.split('.')
            # if F[1] == '-':
            dd = int(F[1:])
            if len(a1) <= dd:
                a1 = '0' * (dd - len(a1) + 1) + a1
            a = a1[:-dd] + '.' + a1[-dd:] + a2
            a = a.rstrip('0')
            # else:
            #     dd = int(F[1:])
            #     if len(a2) < dd:
            #         a2 += '0' * (dd - len(a2))
            #     a = a1 + a2[:dd] + '.' + a2[dd:]
            if a[-1] == '.':
                a = a[:-1]
            return f'{sign}{a}'
        else:
            raise ValueError
        
    if len(V) == 2:
        a, b = V
        a1, a2 = (a+'.').split('.')[:2]
        b1, b2 = (b+'.').split('.')[:2]
        a2, b2 = a2.rstrip('0'), b2.rstrip('0')
        if a2 != '' or b2 != '':
            if F in '+-':
                dd = max(len(a2), len(b2))
                a2 += '0' * (dd - len(a2))
                b2 += '0' * (dd - len(b2))
                a = str(int(a1 + a2))
                b = str(int(b1 + b2))
                return f'(e{dd} ({F} {a} {b}))'
            elif F == '*':
                a = str(int(a1 + a2))
                b = str(int(b1 + b2))
                return f'(e{len(a2)+len(b2)} ({F} {a} {b}))'
            elif F == '/':
                return f'({F} {a} {b})'
            else:
                raise ValueError
        else: # a2 == '' and b2 == ''
            if F in '+-':
                r = eval(f'{a1} {F} {b1}')
                r = str(r)[::-1]
                return f'(f {r})'
                # r = str(r)
                # return f'{r}'
            elif F == '*':
                if len(a1.lstrip('-')) < len(b1.lstrip('-')):
                    return f'({F} {b1} {a1})'
                if (a1[0] == '-') ^ (b1[0] == '-'):
                    sign = '-'
                else:
                    sign = ''
                a1, b1 = a1.lstrip('-'), b1.lstrip('-')
                if len(b1) == 1:
                    if b1 == '0':
                        return '0'
                    else:
                        r = eval(f'{a1} {F} {b1}')
                        r = str(r)[::-1]
                        return f'{sign}(f {r})'
                        # r = str(r)
                        # return f'{sign}{r}'
                else:
                    next_expr = ''
                    for i, bb in enumerate(b1):
                        if bb != '0':
                            next_expr += f' (* {a1} {bb})' + '0' * (len(b1) - i -1)
                    return f'{sign}(+{next_expr})'
                    # next_expr = f'(+ (* {a1} {b1[0]})' + '0' * (len(b1) - 1) + f' (* {a1} {b1[1:]}))'
                    # return f'{sign}{next_expr}'
    else: # len(V) > 2
        if F == '-':
            a = V[0]
            bs = V[1:]
            next_expr = '(+'
            for b in bs:
                next_expr += f' {b}'
            next_expr += ')'
            return f'(- {a} {next_expr})'
        if F in '+*':
            if len(V) % 2 == 1:
                residue = V.pop()
            else:
                residue = None
            next_expr = f'({F}'
            for i in range(len(V)//2):
                a, b = V[i * 2], V[i * 2 + 1]
                next_expr += f' ({F} {a} {b})'
            if residue != None:
                next_expr += f' {residue}'
            next_expr += ')'
            return next_expr

def inference_one_step(expr, only_one=False):
    matches = re.finditer(r'\([^()]+\)', expr)
    cnt = len(list(matches))
    matches = re.finditer(r'\([^()]+\)', expr)
    if cnt == 0:
        if expr == '-0':
            return '0'
        else:
            return expr
    elif cnt >= 1:
        i = 0
        next_expr = ''
        for match in matches:
            next_expr += expr[i:match.start()] + inference_core(expr[match.start():match.end()])
            i = match.end()
            if only_one:
                break
        next_expr += expr[i:]
        return next_expr
    
def inference_step_by_step(expr):
    try:
        chain, next_expr = expr, inference_one_step(expr)
        chain, expr = f'{chain}={next_expr}', next_expr
        while True:
            next_expr = inference_one_step(expr)
            if next_expr != expr:
                chain, expr = f'{chain}={next_expr}', next_expr
            else:
                break
        return chain
    except:
        print('Error:', expr)

def infix_pre_gen(nums_cnt, digit, level=1, integer=True, positive=True, Fs="+-*", father_f=None):
    F = random.choice(list(Fs))
    if father_f == None:
        need_bracket = False
    elif F == "*" and father_f in "+-":
        need_bracket = False
    else:
        need_bracket = True

    if level == 1:
        nums = []
        for i in range(random.randint(2, nums_cnt)):
            num = random.randint(0, 10**random.randint(1, digit))
            if not integer and random.random() < 0.5:
                num /= 10**random.randint(1, digit//2)
            if not positive and random.random() < 0.1:
                num = -num
            nums.append(f"{num:.10f}".rstrip('0').rstrip('.'))
        infix = f"{F}".join(nums)
        if need_bracket:
            infix = f"({infix})"
        pre = f"({F} {' '.join(nums)})"
        return infix, pre
    else:
        infix, pre = [], []
        for i in range(random.randint(2, nums_cnt)):
            if random.random() < 0.5:
                infix_, pre_ = infix_pre_gen(nums_cnt, digit, level-1, integer, positive, Fs, F)
                infix.append(infix_)
                pre.append(pre_)
            else:
                num = random.randint(0, 10**random.randint(1, digit))
                if not integer and random.random() < 0.3:
                    num /= 10**random.randint(1, digit//2)
                if not positive and random.random() < 0.3:
                    num = -num
                infix.append(f"{num:.10f}".rstrip('0').rstrip('.'))
                pre.append(f"{num:.10f}".rstrip('0').rstrip('.'))
        infix = f"{F}".join(infix)
        if need_bracket:
            infix = f"({infix})"
        pre = f"({F} {' '.join(pre)})"
        return infix, pre


# random.seed(0)
# infix, pre = infix_pre_gen(nums_cnt=4, digit=6, level=2, integer=False, positive=False, Fs="+-*")
# str_r = inference_step_by_step(pre)
# print(f"{infix}={str_r}")
# print(len(infix) + len(str_r)) #优化前 2034
# exit()


if __name__ == "__main__":
    datas = []
    cnt = 10000 #生成数据量
    print("start")
    print("step 1/7")
    for i in tqdm(range(0, 10000)):
        infix = f"%d+%d"%(i // 100, i % 100)
        str_r = inference_step_by_step(f"(+ %d %d)"%(i // 100, i % 100))
        datas.append(f"{infix}={str_r}\n")

    print("step 2/7")
    for i in tqdm(range(0, 10000)):
        infix = f"%d*%d"%(i // 100, i % 100)
        str_r = inference_step_by_step(f"(* %d %d)"%(i // 100, i % 100))
        datas.append(f"{infix}={str_r}\n")

    print("step 3/7")
    for i in tqdm(range(cnt)):
        for n in range(2, 20):
            infix, pre = infix_pre_gen(nums_cnt=2, digit=n, level=1, integer=False, positive=False, Fs="+-")
            str_r = inference_step_by_step(pre)
            datas.append(f"{infix}={str_r}\n")
    
    print("step 4/7")
    for i in tqdm(range(cnt)):
        for n in range(2, 20):
            infix, pre = infix_pre_gen(nums_cnt=6, digit=n, level=1, integer=False, positive=False, Fs="+-")
            str_r = inference_step_by_step(pre)            
            datas.append(f"{infix}={str_r}\n")
    
    print("step 5/7")
    for i in tqdm(range(cnt)):
        for n in range(2, 10):
            infix, pre = infix_pre_gen(nums_cnt=2, digit=n, level=1, integer=False, positive=False, Fs="*")
            str_r = inference_step_by_step(pre)
            datas.append(f"{infix}={str_r}\n")

    print("step 6/7")
    for i in tqdm(range(cnt)):
        for n in range(2, 10):
            infix, pre = infix_pre_gen(nums_cnt=4, digit=n, level=1, integer=False, positive=False, Fs="+-*")
            str_r = inference_step_by_step(pre)
            datas.append(f"{infix}={str_r}\n")

    print("step 7/7")
    for i in tqdm(range(cnt)):
        for n in range(2, 6):
            infix, pre = infix_pre_gen(nums_cnt=4, digit=n, level=2, integer=False, positive=False, Fs="+-*")
            str_r = inference_step_by_step(pre)
            datas.append(f"{infix}={str_r}\n")

    random.shuffle(datas)
    # datas = sorted(datas, key=lambda data: len(data))
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    with open(input_file_path, 'w') as f:
        for data in datas:
            f.write(data)

    print('finally')
    
