from collections import defaultdict
import numpy as np
import os
import re
import os.path as op
import pandas as pd
import unicodedata
from sklearn.metrics import f1_score
from arglu.plot_argument_graphs import show_graph

import networkx as nx

# from arglu.file_type_utils import read_textgraph, write_textgraph
# from arglu.graph_processing import make_arg_dicts_from_graph, make_graph_from_arg_dicts, get_perspectives_dict
# from arglu.mutable_tree import MutableTree
# from arglu.node import Node
# from arglu.plot_argument_graphs import show_graph

import re

def get_alphanum(string):
    return re.sub(r'\W+', '', string)


def normalize_list_data(text_list):
    return [get_alphanum(unicodedata.normalize("NFKD", t)) for t in text_list]


def get_leaves(networkx_graph):
    leaves = [node for node in networkx_graph.nodes() 
            if networkx_graph.out_degree(node) == 1 and 
               networkx_graph.in_degree(node) == 0 ]
    leaf_texts = [networkx_graph.nodes()[leaf]["text"] for leaf in leaves]
    return list(sorted(leaf_texts))


def get_leaf_perspectives(leaf_list, networkx_graph):

    nodes, relations = make_arg_dicts_from_graph(networkx_graph)
    text2node = {t: n for n, t in nodes.items()}
    perspective_dict = get_perspectives_dict(nodes, relations)
    text2perspective = {t: perspective_dict[n] for t, n in text2node.items()}

    return [text2perspective[t] for t in leaf_list]

def get_gold_and_pred_perspectives(gold_graph, predicted_graph):

    predicted_leaves = get_leaves(predicted_graph)
    gold_leaves = get_leaves(gold_graph)

    assert normalize_list_data(gold_leaves) == normalize_list_data(predicted_leaves)

    pred_perspectives = get_leaf_perspectives(predicted_leaves, predicted_graph)
    gold_perspectives = get_leaf_perspectives(gold_leaves, gold_graph)

    persp_map = {"red":0, "green":1}

    pred_perspectives = [p for p in map(persp_map.get, pred_perspectives)]
    gold_perspectives = [p for p in map(persp_map.get, gold_perspectives)]

    assert len(gold_perspectives) == len(pred_perspectives)

    return gold_perspectives, pred_perspectives


def leaf_stance_accuracy(gold, predicted):
    
    gold_perspectives, pred_perspectives = get_gold_and_pred_perspectives(gold_graph=gold, 
                                                            predicted_graph=predicted)

    n_correct = sum(np.array(gold_perspectives) == np.array(pred_perspectives))
    n_total = len(gold_perspectives)

    return(n_correct / n_total)


def leaf_stance_f1(gold, predicted):
    
    gold_perspectives, pred_perspectives = get_gold_and_pred_perspectives(gold_graph=gold, 
                                                            predicted_graph=predicted)

    return f1_score(y_true=gold_perspectives, y_pred=pred_perspectives)


def parse_text_to_networkx(text):
    node_re = r"\s*(.+)\s*\(\s*(\S+)\s*(.+)\s*\)\s*(.+)"

    text_lines = text.split("\n")
    G = nx.DiGraph(rankdir="TB")

    G.add_node("main topic", text="")

    colon_trans = str.maketrans("","",":")



    for line in text_lines:
        match_obj = re.match(node_re, line)
        if match_obj:

            node_name, relation, parent, comment = match_obj.groups()
            node_name = node_name.strip().lower()
            relation = relation.strip().lower()
            parent = parent.strip().lower()
            comment = comment.strip().lower()
            
            G.add_node(node_name, text=comment.translate(colon_trans))
            G.add_edge(node_name, parent, label=relation.translate(colon_trans))

    return G

if __name__ == "__main__":
    networkx_graph = parse_text_to_networkx(

        """Comment 1 (attacks main topic): It is worse to actively participate in a death then to simply allow an individual to die

Comment 2 (attacks main topic): The act of killing can wreak immense psychological damage upon rational individuals

Comment 3 (attacks main topic): We should not will a world where killing is acceptable in to existencele in to existence

Comment 4 (supports main topic): A utilitarian approach will result in a decision that saves the largest number of lives possible.

Comment 5 (supports main topic): The human right to life compels us to save as many as possible

Comment 6 (attacks Comment 1): Consequences do in fact matter more."""
    )
    show_graph(networkx_graph, show_perspectives=True)
    print(networkx_graph)

    