from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple

# from nltk.corpus import wordnet


# Constants

# Correcting the errors of the POS tagger for frequent words
NOUN_WORDS = {
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
}


@dataclass
class UDNode:
    """Stores all UD fields under their canonical names according
    to the CoNLL-U format."""
    ID: str  # Word index, integer starting at 1 for each new sentence; may be
    # a range for multiword tokens; may be a decimal number for empty
    # nodes (decimal numbers can be lower than 1 but must be greater
    # than 0).
    FORM: str  # Word form or punctuation symbol.
    LEMMA: str  # Lemma or stem of word form.
    UPOS: str  # Universal part-of-speech tag.
    XPOS: str  # Language-specific part-of-speech tag;
    # underscore if not available.
    FEATS: str  # List of morphological features from the universal feature
    # inventory or from a defined language-specific extension;
    # underscore if not available.
    HEAD: str  # Head of the current word, which is either a value of ID
    # or zero (0).
    # Universal dependency relation to the HEAD (root iff HEAD = 0)
    DEPREL: str
    # or a defined language-specific subtype of one.
    DEPS: str  # Enhanced dependency graph in the form of a list of head-deprel
    # pairs.
    MISC: str  # Any other annotation.

    def __str__(self):
        fields = list(self.__dataclass_fields__)
        return '\t'.join(self.__dict__[f] for f in fields)


@dataclass
class UDEdge:
    head: str
    relation: str
    directionality: str


@dataclass
class UDTree:
    id_lines: List[str]
    keys: List[str]  # The canonical order of keys.
    nodes: Dict[str, UDNode]  # Nodes with UD attributes indexed by keys.
    # Lists of (key, deprel, direction) tuples indexed by keys.
    graph: Dict[str, List[UDEdge]]

    def __str__(self):
        lines = self.id_lines + [str(self.nodes[key]) for key in self.keys]
        return '\n'.join(lines)

    def get_sentence(self) -> str:
        return ' '.join(self.nodes[key].FORM.lower() for key in self.keys)

    def get_node_children(self, node_idx) -> List[str]:
        return [el.head for el in self.graph[node_idx] if el.directionality == "down"]

    def get_real_root(self) -> str:
        # A well-formed UD tree has a virtual root with a single child.
        return self.get_node_children('0')[0]


def conll2graph(record):
    """Converts sentences described using CoNLL-U format
    (http://universaldependencies.org/format.html) to graphs.
    Returns a dictionary of nodes (wordforms and POS tags indexed
    by line numbers) together with a graph of the dependencies encoded
    as adjacency lists of (node_key, relation_label, direction[up or down])
    tuples."""
    id_lines = []
    keys = []  # To preserve the key ordering
    nodes = {}  # Nodes with UD fields
    graph = defaultdict(list)  # Directed graph of dependency relations.
    # The root is indexed with '0'.
    for line in record.splitlines():
        if line.startswith("#"):
            id_lines.append(line)
            continue
        # print(line)
        fields = line.strip("\n").split("\t")
        # print(len(fields))
        assert len(fields) == 10
        key = fields[0]
        keys.append(key)
        ud_node = UDNode(*fields)
        nodes[key] = ud_node
        parent = ud_node.HEAD
        if parent == '_':
            continue
        relation = ud_node.DEPREL
        graph[key].append(UDEdge(
            head=parent,
            relation=relation,
            directionality="up"))
        graph[parent].append(UDEdge(
            head=key,
            relation=relation,
            directionality="down"))
    return (id_lines, keys, nodes, graph)


def conllu2trees(path):
    with open(path, 'r', encoding='utf-8') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [UDTree(*conll2graph(block)) for block in blocks]


def conll2graph_w_POS_filter(record, disallowed_POS, relation_list='everywhere'):
    """Similar to vanilla conll2graph, but takes as input an enhanced UD
    representation with probabilities for all possible UPOS tags, example:

    12 United United GOLD:PROPN|NOUN:2.2641754486412964e-16|PUNCT:8.825103399148
    955e-16|VERB:2.2641754486412964e-16|PRON:2.2641754486412964e-16|ADP:2.264175
    4486412964e-16|DET:1.5953398634622573e-15|PROPN:1.0|ADJ:1.863737435865362e-1
    4|AUX:2.2641754486412964e-16|ADV:2.2641754486412964e-16|CCONJ:2.780324662990
    0386e-15|PART:1.4051247449917929e-15|NUM:2.2641754486412964e-16|SCONJ:2.2641
    754486412964e-16|X:2.2641754486412964e-16|INTJ:5.47092174944889e-16|SYM:1.87
    8780486898714e-15 NNP ...

    disallowed_POS can be either a single disallowed POS tag or a list of them.
    If a node has a disallowed gold POS tag, a non-disallowed one with the
    highest probability will selected.

    Disallowed POS can be disallowed only in particular relations or anywhere.
    """

    if type(disallowed_POS) == str:
        disallowed_POS = [disallowed_POS]

    # A set of nodes participating in relations where
    # the POS should be replaced.
    affected_nodes = set()
    if relation_list != 'everywhere':
        assert type(relation_list) == list
        # Iterate over nodes; add nodes with ingoing relations of the affected
        # type and their parents to affected_nodes.
        for line in record.splitlines():
            if line.startswith("#"):
                continue
            fields = line.strip("\n").split("\t")
            assert len(fields) == 10
            if '.' in fields[0]:  # see below
                continue
            ud_node = UDNode(*fields)
            if ud_node.DEPREL in relation_list:
                affected_nodes.add(ud_node.ID)
                affected_nodes.add(ud_node.HEAD)

    id_lines = []
    keys = []  # To preserve the key ordering
    nodes = {}  # Nodes with UD fields
    graph = defaultdict(list)  # Directed graph of dependency relations.
    # The root is indexed with '0'.
    for line in record.splitlines():
        if line.startswith("#"):
            id_lines.append(line)
            continue
        fields = line.strip("\n").split("\t")
        assert len(fields) == 10
        UPOS_record = fields[3]
        gold_UPOS, *UPOS_probs = UPOS_record.split('|')
        # Nodes with dots in keys do not have POS suggestions, but they are not
        # in the graph anyway.
        if '.' in fields[0]:
            assert('GOLD' not in gold_UPOS)
            fields[3] = gold_UPOS
        else:
            gold_UPOS = gold_UPOS.split(':')[1]
            if gold_UPOS not in disallowed_POS or fields[0] not in affected_nodes:
                fields[3] = gold_UPOS
            else:
                UPOS_scores = []
                for el in UPOS_probs:
                    tag, score = el.split(':')
                    score = float(score)
                    UPOS_scores.append((score, tag))
                UPOS_scores.sort(reverse=True)
                for _, UPOS_tag in UPOS_scores:
                    if UPOS_tag not in disallowed_POS:
                        fields[3] = UPOS_tag
                        break
                else:
                    raise ValueError("All UPOS tags are disallowed.")
                # If the resulting POS tag is weird or if the word is
                # known to be a noun, make it a NOUN.
                if fields[3] != 'ADJ' or fields[1] in NOUN_WORDS:
                    fields[3] = 'NOUN'
        key = fields[0]
        keys.append(key)
        ud_node = UDNode(*fields)
        nodes[key] = ud_node
        parent = ud_node.HEAD
        if parent == '_':
            continue
        relation = ud_node.DEPREL
        graph[key].append(UDEdge(
            head=parent,
            relation=relation,
            directionality="up"))
        graph[parent].append(UDEdge(
            head=key,
            relation=relation,
            directionality="down"))
    return id_lines, keys, nodes, graph


def purge_dotted_nodes(tree: UDTree):
    new_keys = [el for el in tree.keys if '.' not in el]
    new_nodes = {k: v for k, v in tree.nodes.items() if '.' not in k}
    new_graph = {
        k: [el for el in v if '.' not in el.head] for k, v in tree.graph.items() if '.' not in k
    }
    return UDTree(tree.id_lines, new_keys, new_nodes, new_graph)


def split_predicted_upos(upos_string: str) -> Tuple[str, Dict[str, float]]:
    gold_upos_str, *upos_w_probs = upos_string.split('|')
    gold_upos = gold_upos_str.split(':')[1]
    probs = {}
    for prob_guess in upos_w_probs:
        UPOS, prob = prob_guess.split(':')
        probs[UPOS] = float(prob)
    return (gold_upos, probs)


def classify_deprel(deprel):
    if deprel in {
        'nsubj',
        'obj',
        'obl',
        'iobj',
        'nmod',
        'appos'
    }:
        return 'np'
    elif deprel in {
        'csubj',
        'ccomp',
        'advcl',
        'xcomp'
    }:
        return 'sub_clause'
    else:
        return deprel


def get_deprel(node_idx: str, tree: UDTree, separate_neg=True, collapse_categories=False):
    '''
    Reintroduces neg into the set of syntactic relations and/or
    collapses deprels into several classes when asked.
    '''
    node = tree.nodes[node_idx]
    if node.FORM in {'not', 'n’t', "n't"} and separate_neg:  # for English
        return 'neg'
    deprel = node.DEPREL.split(':')[0]
    if deprel != 'advmod':
        if collapse_categories:
            return classify_deprel(deprel)
        else:
            return deprel
    if 'Polarity=Neg' in node.FEATS and separate_neg:  # advmod + Polarity=Neg -> neg
        return "neg"
    else:
        return deprel


# def real_propn(lemma):
#     for synset in wordnet.synsets(lemma):
#         if synset.hypernyms():
#             return False
#     else:
#         return True


if __name__ == '__main__':
    test_record = """# sent_id = panc0.s4
# text = तत् यथानुश्रूयते।
# translit = tat yathānuśrūyate.
# text_fr = Voilà ce qui nous est parvenu par la tradition orale.
# text_en = This is what is heard.
1	तत्	तद्	DET	_	Case=Nom|…|PronType=Dem	3	nsubj	_	Translit=tat|LTranslit=tad|Gloss=it
2-3	यथानुश्रूयते	_	_	_	_	_	_	_	SpaceAfter=No
2	यथा	यथा	ADV	_	PronType=Rel	3	advmod	_	Translit=yathā|LTranslit=yathā|Gloss=how
3	अनुश्रूयते	अनु-श्रु	VERB	_	Mood=Ind|…|Voice=Pass	0	root	_	Translit=anuśrūyate|LTranslit=anu-śru|Gloss=it-is-heard
4	।	।	PUNCT	_	_	3	punct	_	Translit=.|LTranslit=.|Gloss=."""
    ud_tree = UDTree(*conll2graph(test_record))
    from pprint import pprint
    pprint(ud_tree)
    print()
    for n in ud_tree.nodes.values():
        print(n)
