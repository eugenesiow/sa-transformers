import torch
import os
import logging
import numpy as np
from satransformers.transformer_models.roberta_model import RobertaForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
)
from satransformers.classifier_util import (
    InputExample,
    convert_examples_to_features
)

logger = logging.getLogger(__name__)


class SAClassifier:
    def __init__(
            self, model_type, model_name, num_labels=3, use_cuda=True, cuda_device=-1,
    ):
        """
        Initializes a ClassificationModel model.
        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: Default directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """
        model_classes = {
            "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        }

        self.args = {
            "eval_batch_size": 8,
            "cache_dir": "cache_dir/",
            "no_cache": True,
            "max_seq_length": 128,
            "stride": 0.8,
        }
        self.num_labels = num_labels

        config_class, model_class, tokenizer_class = model_classes[model_type]

        self.config = config_class.from_pretrained(model_name, num_labels=num_labels)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.model = model_class.from_pretrained(model_name, config=self.config)

        self.tokenizer = tokenizer_class.from_pretrained(model_name)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        device = self.device
        model = self.model
        args = self.args

        self._move_model_to_device()

        eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(to_predict)]

        eval_dataset = self.load_and_cache_examples(
            eval_examples, evaluate=True, no_cache=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # model_outputs = preds
        preds = np.argmax(preds, axis=1)

        return preds

    def _move_model_to_device(self):
        self.model.to(self.device)

    @staticmethod
    def _get_inputs_dict(batch):
        return {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "token_type_ids": None}

    def load_and_cache_examples(
            self, examples, evaluate=False, no_cache=False, verbose=True
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args["no_cache"]

        output_mode = "classification"

        os.makedirs(self.args["cache_dir"], exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args["cache_dir"],
            "cached_{}_{}_{}_{}_{}".format(
                mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples),
            ),
        )

        if os.path.exists(cached_features_file) and (
                (not args["reprocess_input_data"] and not no_cache)
                or (mode == "dev" and args["use_cached_eval_features"] and not no_cache)
        ):
            features = torch.load(cached_features_file)
            if verbose:
                logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            if verbose:
                logger.info(f" Converting to features started. Cache is not used.")
            features = convert_examples_to_features(
                examples,
                args["max_seq_length"],
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta", "camembert", "xlmroberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                stride=args["stride"],
            )

            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset
