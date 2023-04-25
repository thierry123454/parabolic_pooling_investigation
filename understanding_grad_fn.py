import torch
import graphviz
from graphviz import Source

x = torch.tensor(1., requires_grad=True)
y = 2 * x
l = y - 5

dl = torch.tensor(1.)

back_add = l.grad_fn

dy = back_add(dl)

back_times = back_add.next_functions[0][0]

dx = back_times(dy[0])

back_x = back_times.next_functions[0][0]

back_x(dx[0])

print(x.grad)



# G = graphviz.Digraph(comment="Backpropagation Graph")

# used_ids = {-1}

# def createDAG(fun):
#     global used_ids
#     id = max(used_ids) + 1
#     used_ids.add(id)

#     if not fun:
#         G.node(str(id), "None" + "_" + str(id))
#         return id

#     string = str(fun).split(" ")[0][1:] + "_" + str(id)
#     G.node(str(id), string)

#     print(fun.next_functions)

#     for child_fun in fun.next_functions:
#         G.edge(str(id), str(createDAG(child_fun[0])))
    
#     return id


# createDAG(l.grad_fn)

# s = Source(G.source, filename="simple", format="png")
# s.view()