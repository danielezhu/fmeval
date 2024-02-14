from fmeval.transforms.transform import Transform, Record
from typing import Dict, List
from abc import ABC
import random
import itertools
import numpy as np


class Perturbation(Transform, ABC):

    def __init__(self, seed: int = 5):
        self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)


class ButterFinger(Perturbation):
    """
    Given a text, add keyboard induced typos in randomly selected words.
    Keyboard induced typos are ones where a character is replaced by adjacent characters on the keyboard.

    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: W quick brmwn fox jumps over the lazy dig 10 times.

    Adopted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/butter_fingers_perturbation/transformation.py
    """

    _transform_name = "ButterFinger"

    # Setting default values from NL-Augmenter
    QUERTY_KEY_APPROX: Dict[str, str] = dict()
    QUERTY_KEY_APPROX["q"] = "qwasedzx"
    QUERTY_KEY_APPROX["w"] = "wqesadrfcx"
    QUERTY_KEY_APPROX["e"] = "ewrsfdqazxcvgt"
    QUERTY_KEY_APPROX["r"] = "retdgfwsxcvgt"
    QUERTY_KEY_APPROX["t"] = "tryfhgedcvbnju"
    QUERTY_KEY_APPROX["y"] = "ytugjhrfvbnji"
    QUERTY_KEY_APPROX["u"] = "uyihkjtgbnmlo"
    QUERTY_KEY_APPROX["i"] = "iuojlkyhnmlp"
    QUERTY_KEY_APPROX["o"] = "oipklujm"
    QUERTY_KEY_APPROX["p"] = "plo['ik"

    QUERTY_KEY_APPROX["a"] = "aqszwxwdce"
    QUERTY_KEY_APPROX["s"] = "swxadrfv"
    QUERTY_KEY_APPROX["d"] = "decsfaqgbv"
    QUERTY_KEY_APPROX["f"] = "fdgrvwsxyhn"
    QUERTY_KEY_APPROX["g"] = "gtbfhedcyjn"
    QUERTY_KEY_APPROX["h"] = "hyngjfrvkim"
    QUERTY_KEY_APPROX["j"] = "jhknugtblom"
    QUERTY_KEY_APPROX["k"] = "kjlinyhn"
    QUERTY_KEY_APPROX["l"] = "lokmpujn"

    QUERTY_KEY_APPROX["z"] = "zaxsvde"
    QUERTY_KEY_APPROX["x"] = "xzcsdbvfrewq"
    QUERTY_KEY_APPROX["c"] = "cxvdfzswergb"
    QUERTY_KEY_APPROX["v"] = "vcfbgxdertyn"
    QUERTY_KEY_APPROX["b"] = "bvnghcftyun"
    QUERTY_KEY_APPROX["n"] = "nbmhjvgtuik"
    QUERTY_KEY_APPROX["m"] = "mnkjloik"
    QUERTY_KEY_APPROX[" "] = " "

    def __init__(self, input_text_key: str, perturbation_prob: float, num_perturbations: int = 5, seed: int = 5):
        super().__init__(seed)
        self.seed = seed
        self.perturbation_prob = perturbation_prob
        self.num_perturbations = num_perturbations
        self.input_text_key = input_text_key
        self.perturbed_text_keys = self._generate_perturbed_text_keys()

    def __call__(self, record: Record) -> Record:
        for perturbed_text_key in self.perturbed_text_keys:
            assert perturbed_text_key not in record, \
                f"Perturbed text key {perturbed_text_key} already exists in record {record}"

        perturbed_texts = self.perturb(record[self.input_text_key])
        for key, text in zip(self.perturbed_text_keys, perturbed_texts):
            record[key] = text

        return record

    def _generate_perturbed_text_keys(self) -> List[str]:
        return [
            f"{self._transform_name}({self.input_text_key})[{i}]"
            for i in range(self.num_perturbations)
        ]

    @property
    def output_keys(self):
        return self.perturbed_text_keys

    def perturb(self, text: str) -> List[str]:
        prob_of_typo = int(self.perturbation_prob * 100)
        perturbed_texts = []
        for _ in itertools.repeat(None, self.num_perturbations):
            butter_text = []
            for letter in text:
                lcletter = letter.lower()
                if lcletter not in self.QUERTY_KEY_APPROX.keys():
                    new_letter = lcletter
                else:
                    if random.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = random.choice(self.QUERTY_KEY_APPROX[lcletter])
                    else:
                        new_letter = lcletter
                # go back to original case
                if not lcletter == letter:
                    new_letter = new_letter.upper()
                butter_text.append(new_letter)
            perturbed_texts.append("".join(butter_text))
        return perturbed_texts
