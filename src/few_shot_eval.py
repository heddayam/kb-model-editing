import argparse
from typing import List, Dict, Optional
import logging
import functools
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)


class PromptTooLongError(Exception):
    """Raised when the prompt is too long for LM to process."""

    pass


class Prompt(str):
    def __new__(cls, *args, is_calibration_prompt: bool = False, **kwargs):
        return str.__new__(cls, *args, **kwargs)

    def __init__(self, *args, is_calibration_prompt: bool = False, **kwargs):
        self.is_calibration_prompt = is_calibration_prompt


class GenerationOutput:
    @classmethod
    def init_gpt2(
        cls,
        completion: str,
        logits: torch.FloatTensor,
        perplexity: torch.FloatTensor,
        hidden_states: Optional[torch.FloatTensor] = None,
    ):
        return cls(
            completion=completion,
            probs=F.softmax(logits, dim=0),
            logits=logits,
            perplexity=perplexity,
            hidden_states=hidden_states,
        )

    @classmethod
    def init_gpt3(cls, completion: str, probs: torch.FloatTensor):
        return cls(completion, probs)

    def __init__(
        self,
        completion: str,
        probs: torch.FloatTensor,
        logits: Optional[torch.FloatTensor] = None,
        perplexity: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
    ):

        self.completion = completion
        self.probs = probs
        self.logits = logits
        self.perplexity = perplexity
        self.hidden_states = hidden_states

    def __repr__(self) -> str:
        return f"completion {self.completion}, probs {self.probs}"


def to_device(tensor_dict: Dict[str, torch.Tensor], device: str):
    return {k: v.to(device) for k, v in tensor_dict.items()}


def extract_completion(gpt_output: str):
    return gpt_output.strip()


class DummyCache:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass


class GPT2Wrapper:
    @classmethod
    @functools.cache
    def initialize_model(cls, model_name):
        # reuse models for multiple GPT2Wrapper instances
        return AutoModelForCausalLM.from_pretrained(model_name)

    def __init__(
        self,
        model_name: str,
        cache_module=None,
        labels: List[str] = None,
        batch_size: int = 8,
        calibrate: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.warning(f"Cannot find gpu, setting device to cpu.")
        self.batch_size = batch_size
        self.calibrate = calibrate

        if cache_module is None:
            self.cache_module = DummyCache()
        else:
            self.cache_module = cache_module

        logger.info(f"Setting batch_size={batch_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Initializing {model_name}")
        self.model_name = model_name
        self.model = self.initialize_model(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval().to(self.device)

        label_ids = []
        if labels is not None:
            for label, label_encoded in zip(
                labels,
                self.tokenizer.batch_encode_plus([" " + l for l in labels])[
                    "input_ids"
                ],
            ):
                label_id = label_encoded[0]
                label_str = self.tokenizer.convert_ids_to_tokens(label_id)
                if len(label_encoded) > 1:
                    logger.warning(
                        f"Cannot find matching id for {label}, using prefix {label_str}"
                    )
                label_ids.append(label_id)

        self.labels = labels
        self.label_ids = torch.tensor(label_ids, dtype=torch.long).to(self.device)
        logger.info(f"Labels: {labels}")

    @property
    def embedding_dim(self):
        return self.model.config.hidden_size

    def perplexity(
        self, last_layer: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        all_token_logits = self.model.lm_head(last_layer)
        all_token_nll = -F.log_softmax(all_token_logits, dim=1)[0:-1]
        actual_next_tokens = tokens[1:]
        next_token_nll = all_token_nll.gather(
            dim=1, index=actual_next_tokens.unsqueeze(1)
        )
        perplexity = torch.exp(next_token_nll.mean())
        if perplexity.isnan():
            return torch.tensor(1.0)
        return perplexity

    def complete(
        self,
        prompts: List[Prompt],
        return_hidden_states=False,
        **generation_kwargs: Dict,
    ) -> List[GenerationOutput]:
        batch = self.tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )

        if batch["input_ids"].shape[1] > self.tokenizer.max_len_single_sentence:
            prompt_length = batch["input_ids"].shape[1]
            model_max_length = self.tokenizer.max_len_single_sentence

            raise PromptTooLongError(
                f"prompt length {prompt_length} > "
                f"model_max_length {model_max_length}"
            )

        batch = to_device(batch, self.device)
        input_length = batch["input_ids"].shape[1]
        output = self.model.generate(
            **batch,
            max_length=input_length + 1,
            output_hidden_states=True,
            **generation_kwargs,
        )
        encoded = output.sequences
        decoded = self.tokenizer.batch_decode(encoded[:, input_length:])

        generation_results = []
        logits_all = output.scores[0]

        for i, raw_completion in enumerate(decoded):
            logits = logits_all[i, self.label_ids]
            if self.labels:
                pred = logits.argmax(0)
                completion = self.labels[pred]
            else:
                completion = extract_completion(raw_completion)

            assert output.hidden_states is not None
            # 1 x layers x [batch_size, input_length, hidden_dim]
            last_layer = output.hidden_states[0][-1][i]
            second_to_last_layer = output.hidden_states[0][-2][i]
            last_layer_filtered = last_layer[batch["attention_mask"][i] == 1]
            second_to_last_layer_filtered = second_to_last_layer[
                batch["attention_mask"][i] == 1
            ]

            hidden_states = (
                torch.stack((second_to_last_layer_filtered, last_layer_filtered))
                .contiguous()
                .detach()
                .cpu()
            )

            # compute perplexity
            input_ids_filtered = batch["input_ids"][i, batch["attention_mask"][i] == 1]
            perplexity = self.perplexity(last_layer_filtered, input_ids_filtered)

            generation_results.append(
                GenerationOutput.init_gpt2(
                    completion,
                    logits.detach().cpu(),
                    perplexity=perplexity.detach().cpu(),
                    hidden_states=hidden_states if return_hidden_states else None,
                )
            )

        return generation_results

    def complete_all_with_hidden_states(
        self,
        prompts: List[Prompt],
        calibration_prompts: Optional[List[Prompt]] = None,
        do_calibrate: bool = True,
    ) -> List[GenerationOutput]:
        return self.complete_all(
            prompts,
            calibration_prompts=calibration_prompts,
            do_calibrate=do_calibrate,
            return_hidden_states=True,
        )

    def complete_all(
        self,
        prompts: List[Prompt],
        calibration_prompts: Optional[List[Prompt]] = None,
        do_calibrate: bool = True,
        **additional_kwargs,
    ) -> List[GenerationOutput]:
        generation_kwargs = {
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
            "return_hidden_states": False,
        }
        generation_kwargs.update(additional_kwargs)

        res = [None] * len(prompts)
        uncached = []
        for i, prompt in enumerate(prompts):
            cache_resp = self.cache_module.get(
                model_name=self.model_name, prompt=prompt, **generation_kwargs
            )
            if cache_resp is not None:
                res[i] = cache_resp
            else:
                uncached.append((i, prompt))

        for i in tqdm(range(0, len(uncached), self.batch_size)):
            chunk = uncached[i : i + self.batch_size]
            chunk_prompts = [tup[1] for tup in chunk]
            outputs = self.complete(chunk_prompts, **generation_kwargs)
            for (j, prompt), output in zip(chunk, outputs):
                self.cache_module.set(
                    output,
                    model_name=self.model_name,
                    prompt=prompt,
                    **generation_kwargs,
                )
                res[j] = output

        if self.calibrate and do_calibrate:
            assert len(calibration_prompts) > 0
            cali_outputs = self.complete(calibration_prompts, **generation_kwargs)
            # stack probabilities
            raw_cali_probs = torch.stack([o.probs for o in cali_outputs])
            W = 1.0 / raw_cali_probs.mean(dim=0)
            for o in res:
                probs = o.probs * W
                probs = probs / probs.sum()
                o.probs = probs
                pred = probs.argmax(0)
                o.completion = self.labels[pred]
        return res


class BoolQProcessor:
    def __init__(self, with_context: bool = False):
        self.dataset = load_dataset("super_glue", "boolq")
        self.val_split = None
        self.test_split = self.dataset["validation"]

        self.train_template = (
            "{passage}\n" if with_context else ""
        ) + "Question: {question}\n" "Answer: {label_text}\n\n"
        self.eval_template = (
            "{passage}\n" if with_context else ""
        ) + "Question: {question}\n" "Answer:"

        self.labels = ["False", "True"]
        self.model_kwargs = {"labels": self.labels}

    def convert_example_to_template_fields(self, example):
        label_text = self.labels[example["label"]]
        return {
            "passage": example["passage"],
            "question": example["question"],
            "label_text": label_text,
        }

    @property
    def calibration_examples(self):
        return [
            {field: null_str for field in ["passage", "question"]}
            for null_str in ["N/A", "[MASK]", ""]
        ]

    def create_prompts(self):
        return [
            self.eval_template.format(**example)
            for example in self.dataset["validation"]
        ], [
            self.eval_template.format(**example)
            for example in self.calibration_examples
        ]

    def extract_predictions(
        self,
        outputs: List[GenerationOutput],
    ):

        dataset = self.dataset["validation"]
        if len(outputs) != len(dataset):
            raise Exception(
                f"number of predictions ({len(outputs)}) != "
                f"number of examples ({len(dataset)})"
            )
        preds = []
        labels = []
        class_dist = [0] * len(self.labels)

        for output, test_example in zip(
            outputs, map(self.convert_example_to_template_fields, dataset)
        ):
            for i, label in enumerate(self.labels):
                if output.completion.startswith(label):
                    pred = label
                    class_dist[i] += 1
                    break
            else:
                raise Exception(
                    f"Completion {output.completion} does not match any of the labels {self.labels}"
                )
            preds.append(pred)
            labels.append(test_example["label_text"])
        acc = sum(
            1 if pred == label else 0 for pred, label in zip(preds, labels)
        ) / len(labels)

        return_dict = {
            "acc": acc,
            "preds": preds,
            "labels": labels,
            "class-dist": class_dist,
        }
        return return_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    model = GPT2Wrapper("gpt2-xl", labels=["False", "True"], calibrate=args.calibrate)
    proc = BoolQProcessor()

    prompts, cali_prompts = proc.create_prompts()
    completions = model.complete_all(prompts, calibration_prompts=cali_prompts)
    eval_res = proc.extract_predictions(completions)
    acc = eval_res["acc"]
    preds, labels = eval_res["preds"], eval_res["labels"]
    precision = sum(1 if p == l == "True" else 0 for p, l in zip(preds, labels)) / sum(
        1 if p == "True" else 0 for p in preds
    )

    recall = sum(1 if p == l == "True" else 0 for p, l in zip(preds, labels)) / sum(
        1 if l == "True" else 0 for l in labels
    )
    f1 = 2 / (1 / precision + 1 / recall)
    print("acc: {:.3f}".format(acc))
    print("f1: {:.3f}".format(f1))


if __name__ == "__main__":
    main()
