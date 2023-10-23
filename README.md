# Improving Cross-lingual Transfer through Subtree-aware Word Reordering

## Introduction

This repository contains an implementation of a reordering algorithm
described in the paper [Improving Cross-lingual Transfer through
Subtree-aware Word Reordering](https://arxiv.org/abs/2310.13583).
The algorithm, which is defined in terms of [Universal Dependencies](https://universaldependencies.org/),
is designed to enhance cross-lingual transfer 
by intelligently reordering words in a sentence. This README
provides step-by-step instructions on how to use the algorithm 
effectively.

This repository also contains an implementation of the reordering algorithm by
[Rasooli and Collins (2019)](https://aclanthology.org/N19-1385/).

## Prerequisites

- The reordering algorithms relies on having a UD parser for the 
source language. The package utilizes [Trankit](https://github.com/nlp-uoregon/trankit) 
internally in order to parse the source sentences,
which means, for basic usage, it only supports languages that are supported by Trankit.
For advance usage, it is possible to provide your own parse tree of the sentence.
- The reordering algorithm is based  on pairwise constraints regulating the linear order
of subtrees that share a common parent, which we term POCs for
“pairwise ordering constraints”. For further information about POCs please see the paper.
These POCs can be extracted from UD treebanks and are required for the running of the algorithm.
Several POCs have been extracted and are available for usage. 
For the full list of avilable POCs see the ```estimates_path_dict``` in the `init` function of 
the `UdReorderingAlgo` class in `reordering_package/ud_reorder_algo.py`. To further extract more POCs see the guide below.

## Usage

1. The first step is to define the reordering algorithm type and source language of the sentences to reorder.
````
# Import the reordering algorithm main class
from reordering_package.ud_reorder_algo import UdReorderingAlgo

# Define the input language. If you are relying on Trankit to parse the source sentence, the source language need to be supported by it.
input_language: str = "english"

# Define the reorder algorithm variant. Currently the following algorithms are supported: UdReorderingAlgo.ReorderAlgo.HUJI and UdReorderingAlgo.ReorderAlgo.RASOOLINI.
algorithm_type = UdReorderingAlgo.ReorderAlgo.HUJI

# Get the reorder algorithm main clas.
reorder_algorithm = UdReorderingAlgo(algorithm_type, input_language)
````

2. Define the sentence to be reordered and the target language to reorder by.
````
sentence: str = "The New York Times is a newspaper in the US"
reorder_by_lang: str = "korean"
````

3. (Optional) In some cases, the sentence contain subsequences that must stay intact in order for the
sentence (or annotation) to remain valid (e.g., a proper-name sequence such as The New York Times
may have internal structure but cannot be reordered). These are represented by a list of
dictionaries with the keys 'inclusive_span', with a value of a tuple.
````
entities = [
            {"inclusive_span": (0,3)
           ]
````

4. Get the reordering mapping of the source sentence tokens to the target language word order.
This is an object of type ```reorder_mapping (Dict[int, int])```
which represent a reorder mapping of ```original_pos -> reordered_position``` of a sentence.
```
# Get the reorder mapping.
mapping = reorder_algorithm.get_entities_aware_reorder_mapping(sentence,
                                                               reorder_by_lang,
                                                               entities)
```

5. Reorder the sentence.
````
# Reorder the parsed sentence.
reordered_sentence: str = UdReorderingAlgo.reorder_sentence(sentence, mapping)
````

### Using external UD parse tree
In some cases you might not want to rely on the internal usage of the Trankit parser in order 
to parse the source sentence. Such cases might be, if Trankit does not support the source language,
or if you have a gold (or just better) parse tree of the sentence. In that case it is possible
to supply parse tree to the reordering algorithm in the following way:

1. Prepare the parse tree in the ```TokenList``` format as defined in the [conllu](https://github.com/EmilStenstrom/conllu)
package.
2. Instead of calling ```get_entities_aware_reorder_mapping``` in step 4, call the following function, and proceed as usual.
````
import conllu

parse_tree: conllu.TokenList = [parse tree here]
# Get the reorder mapping.
mapping = reorder_algorithm.get_entities_aware_reorder_mapping_with_parse_tree_input(sentence,
                                                                                     reorder_by_lang,
                                                                                     entities,
                                                                                     parse_tree)
````

## Extracting POCs
In order to extract new POCs and use them in the reordering algorithm, do as follows:

### For the reordered algorithm described in paper:
1. Extract the POCs using the following script:
````
python reordering_package/huji_ud_reordering/compute_estimates.py -i [input ud treebank in conllu format] -o [output-dir]
````

2. Add the POCs path to the ```estimates_path_dict``` in the `init` function of 
the `UdReorderingAlgo` class in `reordering_package/ud_reorder_algo.py`.

### For the reordering algorithm by Rasooli and Collins (2019)
1. Extract the required statistics using the following script:
````
python reordering_package/rasoolini_ud_reorder/reorder_rasoolini.py -i [input ud treebank in conllu format] -o [output-dir]
````
2. Add the POCs path to the ```direction_path_dict``` in the `init` function of 
the `UdReorderingAlgo` class in `reordering_package/ud_reorder_algo.py`.

## Citing

If you use the reordering algorithms in your work, please cite the  paper:

````
@misc{arviv2023improving,
      title={Improving Cross-Lingual Transfer through Subtree-Aware Word Reordering}, 
      author={Ofir Arviv and Dmitry Nikolaev and Taelin Karidi and Omri Abend},
      year={2023},
      eprint={2310.13583},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
````

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors
[Dmitry Nikolaev](https://github.com/macleginn)
