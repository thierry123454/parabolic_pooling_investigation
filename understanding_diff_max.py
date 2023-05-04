import torch
import graphviz
from graphviz import Source

# Case 1
inp = torch.tensor([0.5], requires_grad=True)
x_1 = inp ** 2
x_2 = 2 * inp
y = torch.cat([x_1, x_2])
z = torch.max(y)

# Als 2*x > x**2 (wat het geval is want 2*0.5 = 1 > 0.5**2 = 0.25):
#   1. dz/dy = [0, 1]
#   2. dz/dx_2 = 1
#   3. dz/dx_1 = 0
#   4. dz/d(inp) = 2

print()
print("inp = 0.5 => 2*inp >inp**2:")

dz = torch.tensor(1.)

max_back = z.grad_fn
dy = max_back(dz)

print(f'dz/dy = {dy}')

cat_back = max_back.next_functions[0][0]
dx_1, dx_2 = cat_back(dy)

print(f'dz/dx_1 = {dx_1}')
print(f'dz/dx_2 = {dx_2}')

pow_back = cat_back.next_functions[0][0]
dinp_1 = pow_back(dx_1)

mul_back = cat_back.next_functions[1][0]
dinp_2 = mul_back(dx_2)[0] # Other entry is the 2 in 2*x, useless.

accum_inp_1 = pow_back.next_functions[0][0]
accum_inp_1(dinp_1)
accum_inp_2 = mul_back.next_functions[0][0]
accum_inp_2(dinp_2)

print(f'dz/dinp = {inp.grad}')
inp.grad = torch.tensor([0.], dtype=torch.float32)

# Case 2
inp = torch.tensor([3.], requires_grad=True)
x_1 = inp ** 2
x_2 = 2 * inp
y = torch.cat([x_1, x_2])
z = torch.max(y)

# Als x**2 > 2*x (wat het geval is want 3**2 = 9 > 2 * 3 = 6):
#   1. dz/dy = [1, 0]
#   2. dz/dx_2 = 0
#   3. dz/dx_1 = 1
#   4. dz/d(inp) = 2 * 3 = 6

print()
print("inp = 3 => inp**2 > 2 * inp:")

dz = torch.tensor(1.)

max_back = z.grad_fn
dy = max_back(dz)

print(f'dz/dy = {dy}')

cat_back = max_back.next_functions[0][0]
dx_1, dx_2 = cat_back(dy)

print(f'dz/dx_1 = {dx_1}')
print(f'dz/dx_2 = {dx_2}')

pow_back = cat_back.next_functions[0][0]
dinp_1 = pow_back(dx_1)

mul_back = cat_back.next_functions[1][0]
dinp_2 = mul_back(dx_2)[0] # Other entry is the 2 in 2*x, useless.

accum_inp_1 = pow_back.next_functions[0][0]
accum_inp_1(dinp_1)
accum_inp_2 = mul_back.next_functions[0][0]
accum_inp_2(dinp_2)

print(f'dz/dinp = {inp.grad}')
inp.grad = torch.tensor([0.], dtype=torch.float32)

# Case 3
inp = torch.tensor([2.], requires_grad=True)
x_1 = inp ** 2
x_2 = 2 * inp
y = torch.cat([x_1, x_2])
z = torch.max(y)

# Als x**2 == 2*x (wat het geval is want 2**2 = 4 > 2 * 2 = 4):
#   1. dz/dy = [0.5, 0.5]
#   2. dz/dx_2 = 0.5
#   3. dz/dx_1 = 0.5
#   4. dz/d(inp) = 2 * 2 * 0.5 + 2 * 0.5 = 3

print()
print("inp = 3 => inp**2 > 2 * inp:")

dz = torch.tensor(1.)

max_back = z.grad_fn
dy = max_back(dz)

print(f'dz/dy = {dy}')

cat_back = max_back.next_functions[0][0]
dx_1, dx_2 = cat_back(dy)

print(f'dz/dx_1 = {dx_1}')
print(f'dz/dx_2 = {dx_2}')

pow_back = cat_back.next_functions[0][0]
dinp_1 = pow_back(dx_1)

mul_back = cat_back.next_functions[1][0]
dinp_2 = mul_back(dx_2)[0] # Other entry is the 2 in 2*x, useless.

accum_inp_1 = pow_back.next_functions[0][0]
accum_inp_1(dinp_1)
accum_inp_2 = mul_back.next_functions[0][0]
accum_inp_2(dinp_2)

print(f'dz/dinp = {inp.grad}')
inp.grad = torch.tensor([0.], dtype=torch.float32)

# Maak graaf van gradient functies
G = graphviz.Digraph(comment="Backpropagation Graph")

used_ids = {-1}

def createDAG(fun):
    global used_ids
    id = max(used_ids) + 1
    used_ids.add(id)

    if not fun:
        G.node(str(id), "None" + "_" + str(id))
        return id

    string = str(fun).split(" ")[0][1:] + "_" + str(id)
    G.node(str(id), string)

    # print(fun.next_functions)

    for child_fun in fun.next_functions:
        G.edge(str(id), str(createDAG(child_fun[0])))
    
    return id


createDAG(z.grad_fn)
s = Source(G.source, filename="simple_max", format="png")

# General case
inp_gen = torch.tensor([1.], requires_grad=True)
x_1 = inp_gen**1
x_2 = inp_gen**2
x_3 = inp_gen**3
x_4 = inp_gen**4
x_5 = inp_gen**5
y = torch.cat([x_1, x_2, x_3, x_4, x_5])
z = torch.max(y)
dz = torch.tensor(1.)

#   1. dz/dy = [0.2, 0.2, 0.2, 0.2, 0.2]
#   2. dz/d(inp) = dz/dx_1 * 0.2 + dz/dx_2 * 0.2 + dz/dx_3 * 0.2 + dz/dx_4 * 0.2 + dz/dx_5 * 0.2
#                = 1 * 0.2 + 2 * 1 * 0.2 + 3 * 1^2 * 0.2 + 4 * 1^3 * 0.2 + 5 * 1^4 * 0.2
#                = 3

print()
print("z = max(x**1, x**2, x**3, x**4, x**5)")

max_back = z.grad_fn
dy = max_back(dz)

print(f'dz/dy = {dy}')

cat_back = max_back.next_functions[0][0]
dx_1, dx_2, dx_3, dx_4, dx_5 = cat_back(dy)

print(f'dz/dx_1 = {dx_1}')
print(f'dz/dx_2 = {dx_2}')
print(f'dz/dx_3 = {dx_3}')
print(f'dz/dx_4 = {dx_4}')
print(f'dz/dx_5 = {dx_5}')

pow_back = cat_back.next_functions[0][0]
dinp_1 = pow_back(dx_1)
accum_inp = pow_back.next_functions[0][0]
accum_inp(dinp_1)

pow_back = cat_back.next_functions[1][0]
dinp_2 = pow_back(dx_2)
accum_inp(dinp_2)

pow_back = cat_back.next_functions[2][0]
dinp_3 = pow_back(dx_3)
accum_inp(dinp_3)

pow_back = cat_back.next_functions[3][0]
dinp_4 = pow_back(dx_4)
accum_inp(dinp_4)

pow_back = cat_back.next_functions[4][0]
dinp_5 = pow_back(dx_5)
accum_inp(dinp_5)

print(f'dz/dinp = {inp_gen.grad}')

# Conclusion: gradient to variables that cause max = 1 / n where n is the amount of variables that
# cause a maximum!

# Maak graaf van gradient functies
G = graphviz.Digraph(comment="Backpropagation Graph")

used_ids = {-1}

def createDAG(fun):
    global used_ids
    id = max(used_ids) + 1
    used_ids.add(id)

    if not fun:
        G.node(str(id), "None" + "_" + str(id))
        return id

    string = str(fun).split(" ")[0][1:] + "_" + str(id)
    G.node(str(id), string)

    # print(fun.next_functions)

    for child_fun in fun.next_functions:
        G.edge(str(id), str(createDAG(child_fun[0])))
    
    return id


createDAG(z.grad_fn)
s = Source(G.source, filename="general_max", format="png")
# s.view()