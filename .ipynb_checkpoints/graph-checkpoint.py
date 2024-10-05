from graphviz import Digraph
import os
import random
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"


dot = Digraph() 

#Makes it go the right way
#dot.attr(rankdir='LR')
dot.attr(rankdir='RL'); dot.attr('edge', dir='back') 

dot.attr('node', shape='box')

def graph(tensor, level=0, name=None):
    dot = Digraph() 

    #Makes it go the right way
    dot.attr(rankdir='RL'); dot.attr('edge', dir='back') 

    dot.attr('node', shape='box')
    
    def reccursion(tensor, level=0, name=None):
        if not name:
            name = str(random.randint(0,100)) + str(level)
        if tensor.label:
            dot.node(name, tensor.label)
        else:
            dot.node(name, "Weights or Input")
        
        for p in tensor.parents:
            p_name = str(random.randint(0,100)) + str(level+1)
            dot.node(p_name, tensor.label)
            dot.edge(name, p_name)
            reccursion(p, level=level+1, name=p_name)
    reccursion(tensor)
    return dot