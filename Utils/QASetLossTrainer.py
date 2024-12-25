import torch
from torch.nn.functional import cross_entropy as CrossEntropyLoss, pad
from .computeQASetValidationMetrics import *
from transformers import Seq2SeqTrainer
from .defaultValues import *


# subclass trainer
class QASetLossTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        global DEFAULT_LAMBDA1, DEFAULT_LAMBDA2, DEFAULT_QA_SEP_TOKENS, DEFAULT_Q_SEP_TOKENS, DEFAULT_A_SEP_TOKENS, DEFAULT_PADDING_IDX, DEFAULT_DEVICE

        super().__init__(*args, **kwargs)
        # for each of the values, if exists in kwargs, assign it to the class attribute
        # else assign default value
        keys = kwargs.keys()
        self.LAMBDA1 = kwargs['lambda1'] if keys.__contains__('lambda1') else DEFAULT_LAMBDA1
        self.LAMBDA2 = kwargs['lambda2'] if keys.__contains__('lambda2') else DEFAULT_LAMBDA2
        self.QA_sep_tokens_tensor = kwargs['QA_sep_tokens_tensor'] if keys.__contains__(
            'QA_sep_tokens_tensor') else DEFAULT_QA_SEP_TOKENS
        self.QA_sep_length = len(self.QA_sep_tokens_tensor)
        self.A_sep_tokens_tensor = kwargs['A_sep_tokens_tensor'] if keys.__contains__(
            'A_sep_tokens_tensor') else DEFAULT_A_SEP_TOKENS
        self.A_sep_length = len(self.A_sep_tokens_tensor)
        self.Q_sep_tokens_tensor = kwargs['Q_sep_tokens_tensor'] if keys.__contains__(
            'Q_sep_tokens_tensor') else DEFAULT_Q_SEP_TOKENS
        self.Q_sep_length = len(self.Q_sep_tokens_tensor)
        self.PADDING_IDX = kwargs['PADDING_IDX'] if keys.__contains__('PADDING_IDX') else DEFAULT_PADDING_IDX
        self.DEVICE = kwargs['DEVICE'] if keys.__contains__('DEVICE') else DEFAULT_DEVICE

    def compute_loss(self, model, inputs, num_items_in_batch=16, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        custom_loss = self.custom_loss_function(logits, labels)
        ce_loss = CrossEntropyLoss(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = self.LAMBDA1 * ce_loss + self.LAMBDA2 * custom_loss

        return (loss, outputs) if return_outputs else loss

    def custom_loss_function(self, logits, targets):
        vocab_size = logits.size(-1)
        size_to_pad = targets.size(1)  # will need it for padding
        predictions = torch.max(logits, dim=-1).indices

        pred_split = [
            split_QAs(prediction, self.QA_sep_tokens_tensor, self.Q_sep_tokens_tensor, self.A_sep_tokens_tensor)
            for prediction in predictions]  # split the predictions to questions and answers
        label_split = [split_QAs(label, self.QA_sep_tokens_tensor, self.Q_sep_tokens_tensor, self.A_sep_tokens_tensor)
                       for label in targets]  # split the labels to questions and answers

        # find the QA seperation indices
        qa_start_indices = [find_subsequence_occurrences(prediction, self.QA_sep_tokens_tensor) for prediction in
                            predictions]
        gold_start_indices = [find_subsequence_occurrences(label, self.QA_sep_tokens_tensor) for label in targets]

        total_gold, total_pred = 0, 0
        pred_stack, gold_stack = [], []  # we want to calculate all the QAs in the batch together for better performance
        for b, (label_qas, pred_qas) in enumerate(
                zip(label_split, pred_split)):  # for each example in the batch, could be multiple QA
            label_answers = []  # will hold all the answers of this example labels
            pred_answers = []  # will hold all the answers of this example predictions
            label_ans_ind, pred_ans_ind = 0, 0
            pred_ans_ind_to_qa = {}
            label_ans_ind_to_qa = {}

            for qa_i, (question, answers) in enumerate(label_qas):
                label_answers.extend(answers)
                q, answers = self.get_question_and_answer_from_indices(gold_start_indices[b], targets[b],
                                                                       qa_i)  # return the tokens ids of the question and of all the answers
                for a_i, ans in enumerate(answers):
                    label_ans_ind_to_qa[label_ans_ind] = torch.cat((q, ans))
                    label_ans_ind += 1
            for i, (question, answers) in enumerate(pred_qas):
                pred_answers.extend(answers)
                q, answers = self.get_question_and_answer_from_indices(qa_start_indices[b], logits[b], i,
                                                                       is_logits=True,
                                                                       logits_pred=predictions[b])
                for a_i, ans in enumerate(answers):
                    pred_ans_ind_to_qa[pred_ans_ind] = torch.cat((q, ans))
                    pred_ans_ind += 1

            total_gold += label_ans_ind
            total_pred += pred_ans_ind

            pred_answers, label_answers = pad_lists_with_empty(pred_answers, label_answers)
            IOU_matrix = build_IOU_metrix(pred_answers,
                                          label_answers)  # matrix to compare between groups of answers; if one group is larger it's like adding more empty lists
            row_ind, col_ind = linear_sum_assignment(IOU_matrix, maximize=True)  # .detach().cpu().numpy()

            for r, c in zip(row_ind, col_ind):
                # pad the answers to be at the same length
                if r in pred_ans_ind_to_qa and c in label_ans_ind_to_qa:

                    pred_stack.append(pad(pred_ans_ind_to_qa[r], (0, 0, 0, size_to_pad - len(pred_ans_ind_to_qa[r])),
                                          value=self.PADDING_IDX))
                    gold_stack.append(
                        pad(label_ans_ind_to_qa[c], (0, size_to_pad - len(label_ans_ind_to_qa[c])),
                            value=self.PADDING_IDX))
                elif r in pred_ans_ind_to_qa:
                    pred_stack.append(pad(pred_ans_ind_to_qa[r], (0, 0, 0, size_to_pad - len(pred_ans_ind_to_qa[r])),
                                          value=self.PADDING_IDX))
                    gold_stack.append(
                        torch.full((size_to_pad,), self.PADDING_IDX, device=self.DEVICE,
                                   dtype=pred_ans_ind_to_qa[r].dtype))
                elif c in label_ans_ind_to_qa:
                    pred_stack.append(torch.full((size_to_pad, vocab_size), self.PADDING_IDX, device=self.DEVICE,
                                                 dtype=label_ans_ind_to_qa[c].dtype))
                    gold_stack.append(
                        pad(label_ans_ind_to_qa[c], (0, size_to_pad - len(label_ans_ind_to_qa[c])),
                            value=self.PADDING_IDX))

        gold_stack = torch.stack(gold_stack).type(torch.LongTensor).to(self.DEVICE)
        pred_stack = torch.stack(pred_stack).to(self.DEVICE)

        loss = CrossEntropyLoss(pred_stack.view(-1, vocab_size), gold_stack.view(-1))

        return loss

    def get_question_and_answer_from_indices(self, QA_start_indices, original_values, QA_index, is_logits=False,
                                             logits_pred=None):
        QA_indices = self.get_QA_indices(QA_start_indices, QA_index, len(original_values))

        if len(QA_indices) == 0:  # Return empty tensor if there are no indices
            return torch.tensor([], device=self.DEVICE, dtype=original_values.dtype), []

        # Select either logits or original values based on `is_logits`
        if is_logits:
            if logits_pred is None:
                raise ValueError("logits_pred must be provided if is_logits is True")
            values = logits_pred[QA_indices]
        else:
            values = original_values[QA_indices]

        # Find the question mark index
        qm_index = find_subsequence_occurrences(values, self.Q_sep_tokens_tensor)
        # If no question mark is found, return the full tensor as question
        if not qm_index:
            return torch.tensor(values, device=self.DEVICE, dtype=original_values.dtype), []

        # Compute question indices
        start_index = 0 if QA_index == 0 else self.QA_sep_length
        question_indices = QA_indices[start_index:qm_index[0] + 1]  # +1 to include the question mark in the question

        # Find answer start indices
        ans_start_indices = find_subsequence_occurrences(values, self.A_sep_tokens_tensor)
        ans_start_indices = [qm_index[0] + 1] + ans_start_indices  # Adjust to start after question

        # Compute answer indices using list comprehension (optimized slicing)
        answer_indices = []
        ans_indices_length = len(ans_start_indices)
        for ans_index in range(ans_indices_length):
            if ans_index >= ans_indices_length:
                answer_indices.append([])
                print(
                    f"Unexpected scenario in get_question_and_answer_from_indices when computing answer indices. Received index = {ans_index} when ans_start_indices are:\n{ans_start_indices}")
                continue

            start_index = ans_start_indices[ans_index]
            if ans_index > 0:
                start_index += self.A_sep_length

            if ans_index < ans_indices_length - 1:
                answer_indices.append(QA_indices[start_index:ans_start_indices[ans_index + 1]])
            else:  # ans_index == ans_indices_length - 1
                answer_indices.append(QA_indices[start_index:])

        question_ret = original_values[question_indices]
        answer_ret = [original_values[indices] for indices in answer_indices]

        return question_ret, answer_ret

    @staticmethod
    def get_QA_indices(start_indices, index, max_len, start_index=0):
        if index < 0 or index > len(start_indices):
            return range(0)  # Empty range
        if index == 0:
            end_index = start_indices[0] if start_indices else max_len
            return range(start_index, end_index)
        if index == len(start_indices):
            return range(start_indices[-1], max_len)
        return range(start_indices[index - 1], start_indices[index])
