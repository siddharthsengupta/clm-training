# Here are some imports that we'll need

from haystack.nodes import DensePassageRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore
import torch.distributed as dist

import argparse
import logging
import os, shutil

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)



def train(
        query_model,
        passage_model,
        data_dir,
        train_file,
        test_file,
        save_dir
):
    ## Initialize DPR model
    print("Initialise retriever")
    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(),
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=512,
    )
    # Start training
    print("Start training")
    retriever.train(
        data_dir=data_dir,
        train_filename=train_file,
        dev_filename=test_file,
        test_filename=test_file,
        n_epochs=30,
        batch_size=8,
        learning_rate=1e-5,
        weight_decay=0.1,
        grad_acc_steps=4,
        save_dir=save_dir,
        evaluate_every=500,##
        embed_title=False,
        # n_gpu=3
        # num_positives=1,
        # num_hard_negatives=1,
    )


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--query_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to query model",
    )
    parser.add_argument(
        "--passage_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to passage model",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model will be saved.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The train and test data folder.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="Train file name.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        required=True,
        help="Test file name",
    )
    parser.add_argument(
        "--use_base_model",
        default=True,
        type=bool,
        required=True,
        help="Use base model or pretrained model",
    )
    args = parser.parse_args()

    print("Training")

    if args.use_base_model:
        query_model = 'facebook/dpr-question_encoder-single-nq-base'
        passage_model = 'facebook/dpr-ctx_encoder-single-nq-base'
    else:
        query_model = args.query_model_name_or_path
        passage_model = args.passage_model_name_or_path

    train(query_model=query_model,
          passage_model=passage_model,
          data_dir=args.data_dir,
          train_file=args.train_file,
          test_file=args.predict_file,
          save_dir=args.output_dir)
    print("Training complete")

if __name__ == "__main__":
    # dist.init_process_group('nccl')
    main()
