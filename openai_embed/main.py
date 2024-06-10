# code largely taken from https://github.com/McGill-NLP/weblinx/tree/main/modeling/dmr
import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from langchain_openai import OpenAIEmbeddings
from sentence_transformers.util import cos_sim, dot_score
from weblinx.processing import group_record_to_dict
from weblinx.utils.recs import ungroup_dict_to_records

from weblinx_baseline.preprocessing import build_records_for_single_demo, build_formatters
from grounding.eval import evaluate
from grounding.model import GroundingModel

def run_model_and_update_groups(
    embeddings_model, input_grouped: Dict[Any, List[dict]], sim_method="cos_sim"
):
    if sim_method == "cos_sim":
        sim_func = cos_sim
    elif sim_method == "dot_product":
        sim_func = dot_score
    else:
        raise ValueError(f"Unknown similarity function: {sim_method}")

    for k, group in tqdm(input_grouped.items(), desc="Computing scores"):
        # group = input_grouped[k]
        query = group[0]["query"]
        docs = [r["doc"] for r in group]

        encoded = embeddings_model.embed_documents([query] + docs)

        query_vector, doc_vectors = encoded[0], encoded[1:]
        scores = sim_func(query_vector, doc_vectors).cpu().squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]

        for i, r in enumerate(group):
            r["score"] = scores[i]


def get_ranks_from_scores(scores: Dict[Any, float], starts_at=1) -> Dict[Any, int]:
    """
    Given a dictionary of key -> scores, return a dictionary of key -> ranks.
    """
    # Get sorted keys
    keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    ranks = {k: i + starts_at for i, k in enumerate(keys)}

    return ranks


def verify_queries_are_all_the_same(grouped_records: dict) -> bool:
    """
    Given a dictionary of grouped records, this function verifies that all
    queries are the same within each group.
    """
    for k, v in grouped_records.items():
        first_query = v[0]["query"]
        if not all(r["query"] == first_query for r in v):
            return False
    return True


class OpenAIEmbedGrounding(GroundingModel):
    def preprocess(self, demos):
        format_intent_input, _ = build_formatters()
        input_records: List[dict] = []
        logging.info(f"Number of demos: {len(demos)}. Starting building records.")
        for demo in tqdm(demos, desc="Building input records"):
            demo_records = build_records_for_single_demo(
                demo=demo,
                format_intent_input=format_intent_input,
                max_neg_per_turn=None,
                only_allow_valid_uid=False,
            )
            input_records.extend(demo_records)
        logging.info(f"Completed. Number of input records: {len(input_records)}")

        # Group records by (demo_name, turn_index) pairs
        input_grouped = group_record_to_dict(
            input_records, keys=["demo_name", "turn_index"], remove_keys=False
        )
        # Verify that queries are all the same within each group
        error_msg = "Queries are not all the same within each group"
        assert verify_queries_are_all_the_same(input_grouped), error_msg

        return input_grouped

    def predict_score(self, samples: dict, group_scores: bool = False):
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # default settings
        sim_method = "cos_sim"

        logging.info(f"Using the following similarity method: {sim_method}")

        run_model_and_update_groups(
            embeddings_model, input_grouped=samples, sim_method=sim_method
        )
        logging.info("Completed")

        for group in samples.values():
            scores = {r["uid"]: r["score"] for r in group}
            ranks = get_ranks_from_scores(scores)
            for r in group:
                r["rank"] = ranks[r["uid"]]

        if group_scores:
            input_records = samples
        else:
            input_records = ungroup_dict_to_records(samples)

        return input_records
    
    def predict(self, demos):
        input_grouped = self.preprocess(demos)
        preds = self.predict_score(input_grouped)

        out = []
        for r in preds:
            out.append({
                'uid': r['uid'],
                'score': r['score'],
                'rank': r['rank'],
                'label': r['label']
            })
        return out


if __name__ == "__main__":
    model = OpenAIEmbedGrounding()
    evaluate(model, split="testing", model_name="openai_embed_large")