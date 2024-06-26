# code largely taken from https://github.com/McGill-NLP/weblinx/tree/main/modeling/dmr
import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from langchain_openai import OpenAIEmbeddings
from sentence_transformers.util import cos_sim, dot_score
from weblinx.utils.recs import ungroup_dict_to_records

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


if __name__ == "__main__":
    model = OpenAIEmbedGrounding()
    evaluate(model, split="testing", model_name="openai_embed_large")