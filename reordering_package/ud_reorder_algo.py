import copy
import json
from collections import OrderedDict
from enum import Enum
from typing import Dict, Tuple, List, Union, Optional
import conllu
import trankit
import reordering_package.huji_ud_reordering.UDLib as UDLib
import reordering_package.huji_ud_reordering.ReorderingNew as ReorderingNew
from reordering_package.rasoolini_ud_reorder.reorder_rasoolini import reorder_tree as reorder_tree_rasoolini


def _create_conllu_node(id: int, form: str, lemma: str, upostag: str, xpostag: str, feats: str,
                        head: int, deprel: str, deps: str, misc: str) -> Dict:
    od = OrderedDict()
    od["id"] = id
    od["form"] = form
    od["lemma"] = lemma
    od["upostag"] = upostag
    od["xpostag"] = xpostag
    od["feats"] = feats
    od["head"] = head
    od["deprel"] = deprel
    od["deps"] = deps
    od["misc"] = misc

    return od


class UdReorderingAlgo:
    class ReorderAlgo(Enum):
        HUJI = 0,
        RASOOLINI = 1,
        HUJI_GUROBI = 2,

    def __init__(self, algo_type: ReorderAlgo, input_lang: str = "english"):
        self._algo_type = algo_type

        self._test_reorder_mapping_expansion()

        self._nlp = trankit.Pipeline(lang=input_lang, gpu=True, cache_dir='./cache')
        self._nlp.set_active(input_lang)
        _estimates_dir = "reordering_package/huji_ud_reordering/data/estimates"
        self._estimates_path_dict = {
            "japanese": f'{_estimates_dir}/ja_gsd-ud-train.conllu.estimates.json',
            "korean": f'{_estimates_dir}/ko_gsd-ud-train.conllu.estimates.json',
            "hindi": f'{_estimates_dir}/hi_hdtb-ud-train.conllu.estimates.json',
            "thai": f'{_estimates_dir}/th_pud-ud-test.conllu.estimates.json',
            "german": f'{_estimates_dir}/de_gsd-ud-train.conllu.estimates.json',
            "spanish": f'{_estimates_dir}/es_gsd-ud-train.conllu.estimates.json',
            "french": f'{_estimates_dir}/fr_gsd-ud-train.conllu.estimates.json',
            "indonesian": f'{_estimates_dir}/id_gsd-ud-train.conllu.estimates.json',
            "turkish": f'{_estimates_dir}/tr_imst-ud-train.conllu.estimates.json',
            "vietnamese": f'{_estimates_dir}/vi_vtb-ud-train.conllu.estimates.json',
            "italian": f'{_estimates_dir}/it_isdt-ud-train.conllu.estimates.json',
            "arabic": f'{_estimates_dir}/ar_padt-ud-train.conllu.estimates.json',
            "persian": f'{_estimates_dir}/fa_perdt-ud-train.conllu.estimates.json',
            "dutch": f'{_estimates_dir}/nl_alpino-ud-train.conllu.estimates.json',
            "polish": f'{_estimates_dir}/pl_pdb-ud-train.conllu.estimates.json',
            "portuguese": f'{_estimates_dir}/pt_gsd-ud-train.conllu.estimates.json',
            "russian": f'{_estimates_dir}/ru_gsd-ud-train.conllu.estimates.json',
            "irish": f'{_estimates_dir}/ga_idt-ud-train.conllu.estimates.json',
            "telugu": f'{_estimates_dir}/te_mtg-ud-train.conllu.estimates.json',
            "tamil": f'{_estimates_dir}/ta_ttb-ud-train.conllu.estimates.json',
            "chinese": f'{_estimates_dir}/zh_gsd-ud-train.conllu.estimates.json',
            "hungarian": f'{_estimates_dir}/hu_szeged-ud-train.conllu.estimates.json',
            "thai_part-1": f'{_estimates_dir}/th_pud-ud-test_part-1.conllu.estimates.json',
        }

        _direction_dir = "reordering_package/rasoolini_ud_reorder/data"
        self._direction_path_dict = {
            "hindi": f'{_direction_dir}/hi_hdtb-ud-train.conllu.right_direction_prop.json',
            "thai": f'{_direction_dir}/th_pud-ud-test.conllu.right_direction_prop.json',
            "german": f'{_direction_dir}/de_gsd-ud-train.conllu.right_direction_prop.json',
            "japanese": f'{_direction_dir}/ja_gsd-ud-train.conllu.right_direction_prop.json',
            "korean": f'{_direction_dir}/ko_gsd-ud-train.conllu.right_direction_prop.json',
            "spanish": f'{_direction_dir}/es_gsd-ud-train.conllu.right_direction_prop.json',
            "french": f'{_direction_dir}/fr_gsd-ud-train.conllu.right_direction_prop.json',
            "indonesian": f'{_direction_dir}/id_gsd-ud-train.conllu.right_direction_prop.json',
            "turkish": f'{_direction_dir}/tr_imst-ud-train.conllu.right_direction_prop.json',
            "vietnamese": f'{_direction_dir}/vi_vtb-ud-train.conllu.right_direction_prop.json',
            "italian": f'{_direction_dir}/it_isdt-ud-train.conllu.right_direction_prop.json',
            "arabic": f'{_direction_dir}/ar_padt-ud-train.conllu.right_direction_prop.json',
            "persian": f'{_direction_dir}/fa_perdt-ud-train.conllu.right_direction_prop.json',
            "dutch": f'{_direction_dir}/nl_alpino-ud-train.conllu.right_direction_prop.json',
            "polish": f'{_direction_dir}/pl_pdb-ud-train.conllu.right_direction_prop.json',
            "portuguese": f'{_direction_dir}/pt_gsd-ud-train.conllu.right_direction_prop.json',
            "russian": f'{_direction_dir}/ru_gsd-ud-train.conllu.right_direction_prop.json',
            "irish": f'{_direction_dir}/ga_idt-ud-train.conllu.right_direction_prop.json',
            "telugu": f'{_direction_dir}/te_mtg-ud-train.conllu.right_direction_prop.json',
            "tamil": f'{_direction_dir}/ta_ttb-ud-train.conllu.right_direction_prop.json',
            "hungarian": f'{_direction_dir}/hu_szeged-ud-train.conllu.right_direction_prop.json',
            "thai_part-1": f'{_direction_dir}/th_pud-ud-test_part-1.conllu.right_direction_prop.json'
        }


    def _parse_sentence_into_ud(self, sent: str) -> conllu.TokenList:
        doc = self._nlp(sent.split(" "), is_sent=True)

        conllu_tokens = [_create_conllu_node(node["id"], node["text"], node["lemma"], node["upos"],
                                             "_", node["feats"] if "feats" in node else "_",
                                             node["head"], node["deprel"], "_",
                                             f'ner:{node["ner"]}' if "ner" in node else "_")
                         for node in doc["tokens"]]
        conllu_token_list = conllu.TokenList(conllu_tokens, {})
        conllu_token_list.metadata["text"] = sent if isinstance(sent, str) else " ".join(sent)

        return conllu_token_list

    def get_reorder_mapping(self, sent: str, reorder_by_lang: str) -> Optional[Dict[int, int]]:
        if self._algo_type == UdReorderingAlgo.ReorderAlgo.HUJI:
            return self._get_huji_reorder_mapping_from_str(sent, reorder_by_lang)
        elif self._algo_type == UdReorderingAlgo.ReorderAlgo.HUJI_GUROBI:
            return self._get_huji_reorder_mapping_from_str(sent, reorder_by_lang)
        elif self._algo_type == UdReorderingAlgo.ReorderAlgo.RASOOLINI:
            return self._get_rasoolini_reorder_mapping_from_str(sent, reorder_by_lang)
        else:
            raise NotImplemented()


    def _get_huji_reorder_mapping(self, input_tree: conllu.TokenList,
                                  reorder_by_lang: str) -> Tuple[Optional[Dict[int, int]], Optional[conllu.TokenList]]:
        input_tree_str = input_tree.serialize()
        assert input_tree_str[-1] == "\n"
        input_tree_str = input_tree_str[:-1]
        input_tree: UDLib.UDTree = UDLib.UDTree(*UDLib.conll2graph(input_tree_str))

        with open(self._estimates_path_dict[reorder_by_lang], 'r', encoding='utf-8') as est:
            estimates = json.load(est)

        def _reorder_tree_wrapper(input: Tuple[UDLib.UDTree, int, Dict]):
            # tree: U.UDTree, i: int, estimates: Dict[str, Dict[Tuple[str, str], int]]
            (tree, i, estimates) = input
            try:
                # TODO: Make it configurable
                return ReorderingNew.reorder_tree(tree, estimates,
                                                  separate_neg=False,
                                                  with_gurobi=self._algo_type == self.ReorderAlgo.HUJI_GUROBI,
                                                  preference_threshold=0.7)
            except Exception as e:
                print(f'Reordering failed on tree #{i}: {e}')
                return

        res = _reorder_tree_wrapper((input_tree, 0, estimates))
        reordered_tree, idx_map = res if res is not None else (None, None)
        idx_map = UdReorderingAlgo._str_mapping_to_int(idx_map) if idx_map is not None else None

        return idx_map, reordered_tree

    def _get_huji_reorder_mapping_from_str(self, sent: str, reorder_by_lang: str) -> Optional[Dict[int, int]]:
        parse_tree = self._parse_sentence_into_ud(sent)
        return self._get_huji_reorder_mapping(parse_tree, reorder_by_lang)[0]

    def _get_rasoolini_reorder_mapping(self, input_tree: conllu.TokenList,
                                       reorder_by_lang: str) -> Tuple[Dict[int, int], conllu.TokenList]:
        with open(self._direction_path_dict[reorder_by_lang], 'r', encoding='utf-8') as est:
            direction_dict = json.load(est)

        reordered_tree, idx_map = reorder_tree_rasoolini(input_tree, direction_dict)

        return UdReorderingAlgo._str_mapping_to_int(idx_map), reordered_tree

    def _get_rasoolini_reorder_mapping_from_str(self, sent: str, reorder_by_lang: str) -> Optional[Dict[int, int]]:
        input_tree = self._parse_sentence_into_ud(sent)

        return self._get_rasoolini_reorder_mapping(input_tree, reorder_by_lang)[0]

    @staticmethod
    def reorder_sentence(sentence: str, mapping: Dict[int, int]) -> str:
        return " ".join([sentence.split()[k - 1] for k in mapping.keys() if k != 0])

    # region Utility functions

    @staticmethod
    def _str_mapping_to_int(mapping: Dict[any, any]) -> Dict[int, int]:
        mapping_copy = {}
        for i in mapping.keys():
            mapping_copy[int(i)] = int(mapping[i])

        return mapping_copy

    @staticmethod
    def get_continuous_mapped_span(span: Tuple[int, int], mapping: Dict[int, int]) -> Union[Tuple[int, int], None]:
        # the span is inclusive
        span_idx_list = list(range(span[0], span[1] + 1))
        # The span index start from 0, but the mapping start from 1
        mapped_idx = [mapping[i + 1] - 1 for i in span_idx_list]
        expected = range(min(mapped_idx), max(mapped_idx) + 1)
        res = set(expected) == set(mapped_idx)

        if res:
            return min(expected), max(expected)
        else:
            return None

    @staticmethod
    def _sort_mapping_dict_by_val(mapping: Dict[int, int]) -> Dict[int, int]:
        inverse_temp_mapping = {v: k for k, v in mapping.items()}

        sorted_mapping = {inverse_temp_mapping[k]: k for k in sorted(inverse_temp_mapping.keys())}

        return sorted_mapping

    # endregion

    # region Entity-aware reordering
    @staticmethod
    def _get_index_mapping_from_merged_sentence_to_unmerged_sentence(merged_entities: List[Dict],
                                                                     merged_sentence_length: int
                                                                     ) -> Dict[int, Union[int, Tuple[int, int]]]:
        """
        Returns a mapping from the tokens of a merged sentence to token an unmerged sentence, based on a list
        of merged entities. PAY ATTENTION! The mapping starts from 1 to be compatible with the reorder mapping!
        For example if we have the merged sentence "I love District-9 very much", and the merge entity is "District 9", so:
        The unmerged sentence is "I love District 9 very much" and the mapping from the merged sentence to unmerged one is:
        1-1, 2-2, 3-(3,4), 4-5, 5-6 .

        Parameters:
            merged_entities (List): A list of the entities that has been merged. The entity dictionary must contain the keys 'inclusive_span'.
            merged_sentence_length (int): The length of the merged sentence.
        Returns:
            mapping (Dict[int, int]): Returns a mapping from the tokens of a merged sentence to token an unmerged sentence. The mapping start for 1!
        """
        # Filtering out duplicate entities
        filtered_entities_dict = {int(e['merged_inclusive_span'][0]): e for e in merged_entities}

        res_mapping = dict()
        curr_target_index = 0
        for i in range(merged_sentence_length):
            if i in filtered_entities_dict:
                entity = filtered_entities_dict[i]
                entity_len = entity['inclusive_span'][1] - entity['inclusive_span'][0] + 1
                # we add +1 so the mapping will start from 1
                res_mapping[i + 1] = list(range(curr_target_index + 1, curr_target_index + 1 + entity_len - 1 + 1))
                curr_target_index = curr_target_index + entity_len
            else:
                # we add +1 so the mapping will start from 1
                res_mapping[i + 1] = curr_target_index + 1
                curr_target_index = curr_target_index + 1

        return res_mapping

    @staticmethod
    def _convert_merge_sentence_mapping_to_unmerged_sentence_mapping(merged_entities: List[Dict],
                                                                     merged_sentence_mapping: Dict[int, int]
                                                                     ) -> Dict[int, int]:
        """
        Returns a reorder mapping for the unmerged sentence, based on the mapping of the merged sentence and a list of
        the merged entities.
        For example if we have the merged sentence "I love District-9 very much", and the merge entity is "District 9",
        and the reordered merged sentence is: "very much love I District-9" and the reorder mapping
        of the merged sentence is
        1-4, 2-3, 3-5, 4-1, 5-2, 0-0

       The unmerged reordering mapping is:
       1-4, 2-3, 3-5, 4-6, 5-1, 6-2, 0-0

        Parameters:
            merged_entities (List): A list of the entities that has been merged. The entity dictionary must contain the keys 'inclusive_span'.
            merged_sentence_mapping (Dict[int, int): The reorder mapping for the merge sentence.
        Returns:
            mapping (Dict[int, int]): Returns a reordered mapping for unmerged sentence.
        """
        assert 0 in merged_sentence_mapping, "reorder mapping is expected to include the root mapping of 0->0"
        merged_sentence_len = len(merged_sentence_mapping) - 1
        merged_entities = list({e['inclusive_span']: e for e in merged_entities}.values())
        original_sentence_index_mapping = \
            UdReorderingAlgo._get_index_mapping_from_merged_sentence_to_unmerged_sentence(merged_entities,
                                                                                          merged_sentence_len)

        # The '_get_index_mapping_from_merged_sentence_to_unmerged_sentence' method requires the entities to be with
        # spans that match the sentences. In this case we want to apply it on the reordered sentence mapping,
        # and thus we change the span to account for that.
        merged_entities_w_reordered_spans = []
        for entity in merged_entities:
            # We only map the merged inclusive span, as this is the only mapping we have.
            # IMPORTANT! The inclusive_span thus is *not* the reordered one! We leave it there to
            # be able to calculate the entity length

            # the mapping is original_pos -> reordered_pos and start from 1
            merged_entity_start = merged_sentence_mapping[entity['merged_inclusive_span'][0] + 1] - 1
            merged_entity_end_inclusive = merged_entity_start

            reordered_entity = copy.deepcopy(entity)
            reordered_entity['merged_inclusive_span'] = (merged_entity_start, merged_entity_end_inclusive)

            merged_entities_w_reordered_spans.append(reordered_entity)

        reordered_sentence_index_mapping = \
            UdReorderingAlgo._get_index_mapping_from_merged_sentence_to_unmerged_sentence(
                merged_entities_w_reordered_spans,
                merged_sentence_len)

        adjusted_mapping = {}
        items_to_add = {}
        for k, v in merged_sentence_mapping.items():
            if k == 0:  # the reorder mapping include a zero mapping for the root
                assert v == 0
                adjusted_mapping[0] = 0
                continue

            mapped_k = original_sentence_index_mapping[k]
            mapped_v = reordered_sentence_index_mapping[v]
            # simple mapping - no expanding of merged entities
            if isinstance(mapped_k, int) and isinstance(mapped_v, int):
                adjusted_mapping[mapped_k] = mapped_v
            else:  # mapping of an extended entity. We will add it later, so it won't interfere with the standard
                # mapping
                assert len(mapped_k) == len(mapped_v)
                for k_i, v_i in zip(mapped_k, mapped_v):
                    items_to_add[k_i] = v_i

        adjusted_mapping.update(items_to_add)

        return UdReorderingAlgo._sort_mapping_dict_by_val(adjusted_mapping)

    @staticmethod
    def _expand_merged_sentence(merged_entities: List[Dict], merged_sentence: str) -> str:
        """
        Returns a string of merged sentence with its merged entities unmerged.
        For example if we have the merged sentence "I love District-9 very much", and the merge entity is "District 9", so:
        The unmerged sentence is "I love District 9 very much".

        Parameters:
            merged_entities (List): A list of the entities that has been merged. The entity dictionary must contain the keys 'inclusive_span' and 'merged_inclusive_span'.
            merged_sentence (str): The merged sentence.
        Returns:
            unmerged_sentence (str): A string of the unmerged sentence.
        """
        # Filtering out duplicate entities
        filtered_entities = {int(e['inclusive_span'][0]): e for e in merged_entities}.values()

        # We sort the entities, so it is easier to expand the sentence
        sorted_entities = sorted(filtered_entities, key=lambda e: e['inclusive_span'][0])

        sentence = merged_sentence
        span_offset = 0  # the span offset is by how much the sentence length was extended due too splitting
        for entity in sorted_entities:
            split_sentence = sentence.split()
            start, end = entity['merged_inclusive_span'][0] + span_offset, entity['merged_inclusive_span'][
                1] + span_offset
            entity_text = split_sentence[start:end + 1]
            assert len(entity_text) == 1
            entity_text = entity_text[0]

            prefix = " ".join(split_sentence[:start])
            split_entity = entity_text.split("-")
            split_entity_text = " ".join(split_entity)
            postfix = " ".join(split_sentence[end + 1:])

            sentence = f'{prefix} {split_entity_text} {postfix}'

            if len(prefix) == 0:
                sentence = sentence[1:]
            if len(postfix) == 0:
                sentence = sentence[:-1]

            entity_text_len = len(split_entity)
            span_offset = span_offset + entity_text_len - 1

            assert entity["inclusive_span"][0] == start
            assert entity["inclusive_span"][1] == start + entity_text_len - 1

        return sentence

    @staticmethod
    def _test_reorder_mapping_expansion():
        # region simple test
        sentence = "I love District 9 in the United States very much"
        merged_sentence = "I love District-9 in the United-States very much"

        reordered_merged_sentence = "in the United-States very much love I District-9"
        merged_sentence_reorder_mapping = UdReorderingAlgo._sort_mapping_dict_by_val(
            {0: 0, 1: 7, 2: 6, 3: 8, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5})
        assert UdReorderingAlgo.reorder_sentence(merged_sentence,
                                                 merged_sentence_reorder_mapping) == reordered_merged_sentence

        reordered_sentence = "in the United States very much love I District 9"
        sentence_reorder_mapping = UdReorderingAlgo._sort_mapping_dict_by_val(
            {0: 0, 1: 8, 2: 7, 3: 9, 4: 10, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6})
        assert UdReorderingAlgo.reorder_sentence(sentence, sentence_reorder_mapping) == reordered_sentence

        entities_to_fix = [
            {
                'text': "District 9",
                'inclusive_span': (2, 3),
                'merged_inclusive_span': (2, 2)
            },
            {
                'text': "District 9",
                'inclusive_span': (2, 3),
                'merged_inclusive_span': (2, 2)
            },
            {
                'text': "United States",
                'inclusive_span': (6, 7),
                'merged_inclusive_span': (6, 6)
            }
        ]

        adjusted_mapping = UdReorderingAlgo._convert_merge_sentence_mapping_to_unmerged_sentence_mapping(
            entities_to_fix,
            merged_sentence_reorder_mapping)

        assert adjusted_mapping == sentence_reorder_mapping

        # endregion

        # region multiple long entities
        sentence = f'The National Unified USSD Platform ( NUUP ), also known as the * 99 # service , ' \
                   f'is a platform that provides access to the Unified Payment Interface ( UPI ) ' \
                   f'service over the USSD protocol .'
        merged_sentence = f'The National-Unified-USSD-Platform ( NUUP ), also known as the * 99 # service , ' \
                          f'is a platform that provides access to the Unified-Payment-Interface ( UPI ) service over ' \
                          f'the USSD protocol .'

        reordered_merged_sentence = f'The the 99 # * service as also known National-Unified-USSD-Platform ( NUUP ), , ' \
                                    f'is a the USSD protocol over that the Unified-Payment-Interface ( ) UPI service ' \
                                    f'to access provides . platform'
        merged_sentence_reorder_mapping = UdReorderingAlgo._sort_mapping_dict_by_val(
            {1: 1, 9: 2, 11: 3, 12: 4, 10: 5, 13: 6, 8: 7, 6: 8, 7: 9, 2: 10, 3: 11, 4: 12, 5: 13, 14: 14, 15: 15,
             16: 16, 29: 17, 30: 18, 31: 19, 28: 20, 18: 21, 22: 22, 23: 23, 24: 24, 26: 25, 25: 26, 27: 27, 21: 28,
             20: 29, 19: 30, 32: 31, 17: 32, 0: 0})
        assert UdReorderingAlgo.reorder_sentence(merged_sentence,
                                                 merged_sentence_reorder_mapping) == reordered_merged_sentence

        reordered_sentence = f'The the 99 # * service as also known National Unified USSD Platform ( NUUP ), , ' \
                             f'is a the USSD protocol over that the Unified Payment Interface ( ) UPI service to ' \
                             f'access provides . platform'
        # 2-> 10 first entity. to 2-> 10,11,1,13, +3 to all key > 2 and values > 10
        # 23-> 23 second entity. 23-> 23,24,25. +2 to 24+ indexes on both side
        sentence_reorder_mapping = UdReorderingAlgo._sort_mapping_dict_by_val(
            {1: 1, 12: 2, 14: 3, 15: 4, 13: 5, 16: 6, 11: 7, 9: 8, 10: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15,
             8: 16, 17: 17, 18: 18, 19: 19, 34: 20, 35: 21, 36: 22, 33: 23, 21: 24, 25: 25, 26: 26, 27: 27, 28: 28,
             29: 29, 31: 30, 30: 31, 32: 32, 24: 33, 23: 34, 22: 35, 37: 36, 20: 37, 0: 0})
        assert UdReorderingAlgo.reorder_sentence(sentence, sentence_reorder_mapping) == reordered_sentence

        entities_to_fix = [
            {
                'text': 'National Unified USSD Platform',
                'inclusive_span': (1, 4),
                'merged_inclusive_span': (1, 1)
            },
            {
                'text': 'Unified Payment Interface',
                'inclusive_span': (25, 27),
                'merged_inclusive_span': (22, 22)
            }
        ]

        adjusted_mapping = UdReorderingAlgo._convert_merge_sentence_mapping_to_unmerged_sentence_mapping(
            entities_to_fix,
            merged_sentence_reorder_mapping)

        assert adjusted_mapping == sentence_reorder_mapping

        # endregion

    @staticmethod
    def _merge_entities_in_sentence(entities_to_merge: List, sentence: str) -> str:
        # TODO: Add docsting?
        # Filtering out duplicate entities
        filtered_entities = {int(e['inclusive_span'][0]): e for e in entities_to_merge}.values()
        # We sort the entities, so it will easier to calculate the span offset
        sorted_entities = sorted(filtered_entities, key=lambda e: e['inclusive_span'][0])

        merged_sentence = sentence
        span_offset = 0
        for entity in sorted_entities:
            split_sentence = merged_sentence.split()
            start, end = entity['inclusive_span'][0] - span_offset, entity['inclusive_span'][1] - span_offset
            entity_text = split_sentence[start:end + 1]
            entity_text_len = len(entity_text)

            prefix = " ".join(split_sentence[:start])
            merged_entity = "-".join(entity_text)
            postfix = " ".join(split_sentence[end + 1:])

            merged_sentence = f'{prefix} {merged_entity} {postfix}'

            if len(prefix) == 0:
                merged_sentence = merged_sentence[1:]
            if len(postfix) == 0:
                merged_sentence = merged_sentence[:-1]

            span_offset = span_offset + entity_text_len - 1

            entity["merged_inclusive_span"] = (start, start)

        return merged_sentence

    def get_entities_aware_reorder_mapping(self, sentence: str,
                                           reorder_by_lang: str,
                                           entities: List,
                                           forced_entities_to_fix=None) -> Optional[Dict[int, int]]:
        """
        Returns a reorder mapping of original_pos -> reordered_position of a sentence, that match the 'reorder_by_lang',
        while keeping the entities in the entity list as a continues unit in the reordered sentence.

        Parameters:
            sentence (str): The sentence to reorder.
            reorder_by_lang (str): The language to reorder by. Must be available in the estimate dir.
            entities (List): A list of the entities that has been merged. The entity dictionary must contain the keys 'inclusive_span'.
            forced_entities_to_fix (List): entities we transform to single token even if they were reordered correctly
        Returns:
            reorder_mapping (Dict[int, int]):  A reorder mapping of original_pos -> reordered_position of a sentence. Can be used with 'reorder_sentence()'.
        """
        if forced_entities_to_fix is None:
            forced_entities_to_fix = list()
        mapping = self.get_reorder_mapping(sentence, reorder_by_lang)

        if mapping is None:
            return None

        entities_to_fix = []
        for entity in entities:
            span = UdReorderingAlgo.get_continuous_mapped_span(entity['inclusive_span'], mapping)
            if span is None:
                entities_to_fix.append(entity)

        entities_to_fix = entities_to_fix + forced_entities_to_fix

        # Removing duplicates
        entities_to_fix = list({ent['inclusive_span']: ent for ent in entities_to_fix}.values())

        # for entity in entities_to_fix:
        #     if "-" in entity['surfaceform']:
        #         print(f'Current algorithm does not support fixing entities with dashes (-)!')
        #         return None

        if len(entities_to_fix) > 0:
            merged_sentence = UdReorderingAlgo._merge_entities_in_sentence(entities_to_fix, sentence)
            merged_sentence_mapping = self.get_reorder_mapping(merged_sentence, reorder_by_lang)

            mapping = UdReorderingAlgo._convert_merge_sentence_mapping_to_unmerged_sentence_mapping(
                entities_to_fix, merged_sentence_mapping)

        # For the rare case an entity that was reordered correctly by the original algorithm, get reordered incorrectly
        # due to our entity-aware-hack.
        # This also a useful sanity-check that our reordering was valid.
        for entity in entities:
            span = UdReorderingAlgo.get_continuous_mapped_span(entity['inclusive_span'], mapping)
            if span is None:
                if entity in forced_entities_to_fix:
                    raise RecursionError("The entity is not getting fixed and the function is getting into"
                                         " endless recursion")
                return self.get_entities_aware_reorder_mapping(sentence, reorder_by_lang, entities,
                                                               [entity] + forced_entities_to_fix)

        try:
            reordered_sentence = self.reorder_sentence(sentence, mapping)
        except Exception as e:
            raise Exception(print(e))
            # return None

        return mapping

    # endregion

    def get_reorder_mapping_with_parse_tree_input(self, sent: str, reorder_by_lang: str,
                                                  parse_tree: conllu.TokenList) -> Optional[Dict[int, int]]:
        if self._algo_type == UdReorderingAlgo.ReorderAlgo.HUJI:
            return self._get_huji_reorder_mapping(parse_tree, reorder_by_lang)[0]
        elif self._algo_type == UdReorderingAlgo.ReorderAlgo.HUJI_GUROBI:
            return self._get_huji_reorder_mapping(parse_tree, reorder_by_lang)[0]
        elif self._algo_type == UdReorderingAlgo.ReorderAlgo.RASOOLINI:
            return self._get_rasoolini_reorder_mapping(parse_tree, reorder_by_lang)[0]
        else:
            raise NotImplemented()
    def get_entities_aware_reorder_mapping_with_parse_tree_input(
            self,
            sentence: str,
            reorder_by_lang: str,
            entities: List,
            parse_tree: conllu.TokenList,
            forced_entities_to_fix=None) -> Optional[Dict[int, int]]:
        """
        Returns a reorder mapping of original_pos -> reordered_position of a sentence, that match the 'reorder_by_lang',
        while keeping the entities in the entity list as a continues unit in the reordered sentence.

        Parameters:
            sentence (str): The sentence to reorder.
            reorder_by_lang (str): The language to reorder by. Must be available in the estimate dir.
            entities (List): A list of the entities that has been merged. The entity dictionary must contain the keys 'inclusive_span'.
            forced_entities_to_fix (List): entities we transform to single token even if they were reordered correctly
        Returns:
            reorder_mapping (Dict[int, int]):  A reorder mapping of original_pos -> reordered_position of a sentence. Can be used with 'reorder_sentence()'.
        """
        if forced_entities_to_fix is None:
            forced_entities_to_fix = list()
        mapping = self.get_reorder_mapping_with_parse_tree_input(sentence, reorder_by_lang, parse_tree)

        if mapping is None:
            return None

        entities_to_fix = []
        for entity in entities:
            span = UdReorderingAlgo.get_continuous_mapped_span(entity['inclusive_span'], mapping)
            if span is None:
                entities_to_fix.append(entity)

        entities_to_fix = entities_to_fix + forced_entities_to_fix

        # Removing duplicates
        entities_to_fix = list({ent['inclusive_span']: ent for ent in entities_to_fix}.values())

        # for entity in entities_to_fix:
        #     if "-" in entity['surfaceform']:
        #         print(f'Current algorithm does not support fixing entities with dashes (-)!')
        #         return None

        if len(entities_to_fix) > 0:
            merged_sentence = UdReorderingAlgo._merge_entities_in_sentence(entities_to_fix, sentence)
            merged_sentence_mapping = self.get_reorder_mapping(merged_sentence, reorder_by_lang)

            mapping = UdReorderingAlgo._convert_merge_sentence_mapping_to_unmerged_sentence_mapping(
                entities_to_fix, merged_sentence_mapping)

        # For the rare case an entity that was reordered correctly by the original algorithm, get reordered incorrectly
        # due to our entity-aware-hack.
        # This also a useful sanity-check that our reordering was valid.
        for entity in entities:
            span = UdReorderingAlgo.get_continuous_mapped_span(entity['inclusive_span'], mapping)
            if span is None:
                if entity in forced_entities_to_fix:
                    raise RecursionError("The entity is not getting fixed and the function is getting into"
                                         " endless recursion")
                return self.get_entities_aware_reorder_mapping(sentence, reorder_by_lang, entities,
                                                               [entity] + forced_entities_to_fix)

        try:
            reordered_sentence = self.reorder_sentence(sentence, mapping)
        except Exception as e:
            raise Exception(print(e))
            # return None

        return mapping
