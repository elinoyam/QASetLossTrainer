import json
from datasets import GeneratorBasedBuilder, DatasetInfo, SplitGenerator, Split
from datasets.features import features, Value, Sequence, Features

_DESCRIPTION = "This is QASrl dataset."
# _URL = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/"
_URL = "./"  # "./datasets/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json",
}


class QASrlDataset(GeneratorBasedBuilder):

    def _info(self):
        return DatasetInfo(
            description=_DESCRIPTION,
            # This is the features of the original dataset
            # features=features.FeatureDict({
            #             "id": Value('string'),
            #             "sentence_id": Value('string'),
            #             "tokenized": {
            #                 "sentence": Value('string'),
            #                 "question": Sequence(Value('string')),
            #                 "answers": {
            #                     "text": Sequence(Value('string')),
            #                     "answer_start_token": Sequence(Value('int32')),
            #                     "answer_start_char": Sequence(Value('int32')),
            #                 },
            #                 "predicate_idx_token": Value('int32'),
            #                 "predicate_idx_char": Value('int32'),
            #                 "predicate_idx_char_end": Value('int32'),
            #             },
            #             "detokenized": {
            #                 "sentence": Value('string'),
            #                 "question": Value('string'),
            #                 "answers": {
            #                     "text": Sequence(Value('string')),
            #                     "answer_start_token": Sequence(Value('int32')),
            #                     "answer_start_char": Sequence(Value('int32')),
            #                 },
            #                 "predicate_idx_token": Value('int32'),
            #                 "predicate_idx_char": Value('int32'),
            #                 "predicate_idx_char_end": Value('int32'),
            #             },
            #             "predicate": Value('string'),
            #             "question_slots": Sequence(Value('string')),
            #             "is_verbal": Value('bool'),
            #             "verb_form": Value('string'),
            #         }),

            # This is the features of the new dataset
            features=Features({
                "id": Value('int32'),
                "sentence_id": Value('string'),
                "sentence": Value('string'),
                "predicate": Value('string'),
                "qa": Value('string'),
                # "answers": Sequence(Value('string')),
            }),
        )

    def _split_generators(self, dl_manager):
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
            # TODO: combine the q&a of each sentence_id together as a single example - need to check
            hold = {}
            for example in data:
                id = example["id"]
                sentence_id = example["sentence_id"]
                detokenized = example["detokenized"]
                detokenized_sentence = detokenized["sentence"]
                detokenized_question = detokenized["question"] + " "
                detokenized_answers = detokenized["answers"]["text"]
                predicat = example["predicate"]
                if sentence_id not in hold:
                    hold[sentence_id] = {"sentence_id": sentence_id, "sentence": detokenized_sentence}
                if predicat not in hold[sentence_id]:
                    hold[sentence_id][predicat] = []  # { "predicate": predicat, "qa": []}
                hold[sentence_id][predicat].append((detokenized_question+" <A> ".join(detokenized_answers)))

            k = 0
            for sentence_id, value in hold.items():
                for predicate, qa in value.items():
                    # skip sentence_id and sentence
                    if predicate in ["sentence_id", "sentence"]:
                        continue
                    yield k, {
                        "id": k,
                        "sentence_id": sentence_id,
                        "sentence": value["sentence"],
                        "predicate": predicate,
                        "qa": " <QA> ".join(qa)
                    }
                    k += 1

                # yield id, {
                #     "id": id,
                #     "sentence_id": sentence_id,
                #     "sentence": detokenized_sentence,
                #     "question": detokenized_question,
                #     "answers": detokenized_answers,
                #     "predicate": predicat,
                # }
