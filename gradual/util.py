import numpy as np
from graphviz import Digraph
from .engine import Value, Tensor

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        if isinstance(n, Value):
            data = f'{n.data:.4f}'
            grad = f'{n.grad:.4f}'
        else:
            data = np.array2string(n.data, precision=4, separator=",")
            grad = np.array2string(n.grad, precision=4, separator=",")
        node_label = f'{{ {n.label} | data {data} | grad {grad} }}'
        dot.node(name=str(id(n)), label = node_label, shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
