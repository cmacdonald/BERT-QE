from functions import model_fn_builder
import tensorflow as tf
from pyterrier.transformer import TransformerBase
from pyterrier.model import add_ranks
import numpy as np
from bert import tokenization
def convert_query_example(query_id, query, max_seq_length, tokenizer):
    
    query = tokenization.convert_to_unicode(query)

    query_tokens = tokenization.convert_to_bert_input(
        text=query,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        add_cls=True,
        add_sep=True)

    return query_tokens

def convert_single_piece_example(
        ex_index, query_token_ids, 
        piece, max_seq_length,
        tokenizer):
    
    piece_tokens = tokenization.convert_to_bert_input(
        text=tokenization.convert_to_unicode(piece),
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        add_cls=False,
        add_sep=True)
    
    features = {
        'query_token_ids': query_token_ids,
        'label': 0,
        'piece_token_ids': piece_tokens}
    
    return features

def input_fn_builder(features, max_qlen, max_dlen, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    assert max_qlen + max_dlen <= seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)
        
        dataset = tf.data.Dataset.from_tensor_slices({
            "query_token_ids" : tf.constant(
                np.stack([
                    np.pad(record["query_token_ids"], (0,max_qlen - len(record["query_token_ids"]) ) ) for record in features]), 
                shape=[num_examples, max_qlen],
                dtype=tf.int64),
            "piece_token_ids" : tf.constant(
                np.stack([
                    np.pad(record["piece_token_ids"], (0,max_dlen - len(record["piece_token_ids"]))) for record in features]), 
                shape=[num_examples, max_dlen],
                dtype=tf.int64),
            "label" : tf.constant(
                [record["label"] for record in features], shape=[num_examples],
                dtype=tf.int64),
            "qc_score" : tf.constant([0.0], shape=[num_examples], dtype=tf.float64)
        })

        def extract_fn(data_record):
            params = ["qc_scores"]
            features = {
                "query_token_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "piece_token_ids": tf.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "label": tf.FixedLenFeature([], tf.int64)
            } #1

            if "qc_scores" in params:
                features["qc_score"] = tf.FixedLenFeature([], tf.float32)

            #sample = tf.parse_single_example(data_record, features)
            #print(sample)

            a_token_ids = tf.cast(data_record["query_token_ids"], tf.int32)
            b_token_ids = tf.cast(data_record["piece_token_ids"], tf.int32)
            label_ids = tf.cast(data_record["label"], tf.int32)

            input_ids = tf.concat((a_token_ids, b_token_ids), 0)

            a_segment_id = tf.zeros_like(a_token_ids)
            b_segment_id = tf.ones_like(b_token_ids)
            segment_ids = tf.concat((a_segment_id, b_segment_id), 0)

            input_mask = tf.ones_like(input_ids)

            features_dict = {
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
                "label_ids": label_ids
            } #2

            if "qc_scores" in params:
                features_dict["qc_score"] = tf.cast(data_record["qc_score"], tf.float32)

            return features_dict

        #dataset = tf.data.TFRecordDataset([dataset_path])

        if is_training:
            dataset = dataset.shuffle(buffer_size=params["train_examples"], reshuffle_each_iteration=True,
                                    seed=1234).repeat(params["num_train_epochs"])
        #print("Spec before conversion: %s" % str(dataset.element_spec))
        dataset = dataset.map(extract_fn)
        #print("Spec after conversion: %s" % str(dataset.element_spec))
        #dataset = dataset.map(extract_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        #    buffer_size=tf.data.experimental.AUTOTUNE)

        padded_shapes_dict = {
            "input_ids": [seq_length],
            "segment_ids": [seq_length],
            "input_mask": [seq_length],
            "label_ids": []
        }

        padding_values_dict = {
            "input_ids": 0,
            "segment_ids": 0,
            "input_mask": 0,
            "label_ids": 0
        }

        if "qc_scores" in params:
            padded_shapes_dict["qc_score"] = []
            padding_values_dict["qc_score"] = 0.0

        #print(padded_shapes_dict)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes_dict,
            padding_values=padding_values_dict,
            drop_remainder=drop_remainder)
        #print(dataset.element_spec)
        return dataset

    return input_fn

class BERTQE(TransformerBase):
    def __init__(self, bert_json_path, checkpoint_path, verbose=False, body_attr="text"):
        self.max_qlen = 128
        self.max_dlen = 256
        self.max_seq_len = 384
        self.body_attr = body_attr
        self.verbose = verbose

        from bert import modeling, tokenization
        bert_config = modeling.BertConfig.from_json_file(bert_json_path)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file="./bert/vocab.txt", do_lower_case=True)
        
        tpu_cluster_resolver = None
        iterations_per_loop = 500
        init_checkpoint = None  # @param {type:"string"}
        use_tpu = False 
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            keep_checkpoint_max=1,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                per_host_input_for_training=is_per_host))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=2,
            init_checkpoint=init_checkpoint,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu)

        
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=model_fn,
            config=run_config,
            eval_batch_size=32,
            predict_batch_size=32,
            params={"qc_scores": "qc_scores"})

        print("BERTQE Ready")

    def _get_scores(self, estimator_results):
        # this method attempts to reproduce the logic from
        # https://github.com/zh-zheng/BERT-QE/blob/master/expansion_inference.py#L172
        
        import numpy as np
        from scipy.special import softmax

        qc_scores = []
        probs = []
        total = 0
        for item in estimator_results:
            qc_scores.append(item["qc_scores"])
            probs.append(item["probs"] )
            total += 1
        probs = np.array(probs)
        cp_scores = np.stack(probs)[:, 1]
        qc_scores = softmax(qc_scores, axis=-1)
        #print(qc_scores)
        #I think this line summed the scores for each passage in a documents, we dont need this
        #scores = np.sum(np.multiply(qc_scores, cp_scores), axis=-1, keepdims=False)
        scores = np.multiply(qc_scores, cp_scores)
        assert len(scores) == total, (len(scores, total))
        return scores
    
    def _convert_query_and_docs_to_features(self, query_id, query, doc_texts):
        query_toks = convert_query_example(query_id, query, self.max_qlen, self.tokenizer)

        features = []
        from pyterrier import tqdm, started
        assert started()
        for (ex_index, example) in enumerate( doc_texts):
            feature = convert_single_piece_example(ex_index, query_toks, example, self.max_dlen, self.tokenizer)

            features.append(feature)
        return features

    def _for_each_query(self, qid, query, document_set):

        # apply the BERT tokeniser
        features = self._convert_query_and_docs_to_features(qid, query, document_set[self.body_attr])
        
        # build examples from the array, no need to save to disk
        input_fn = input_fn_builder(features, self.max_qlen, self.max_dlen, self.max_seq_len, False, False)

        results = self.estimator.predict(input_fn=input_fn,
            yield_single_examples=True,
            checkpoint_path="/users/tr.craigm/projects/pyterrier/BERT-QE/robust04_base/Fold1/model.ckpt-9375")
        
        scores = self._get_scores(results)
        assert len(scores) == len(document_set), (len(scores) , len(document_set))
        return scores
        
    def transform(self, topics_and_res):
        import pandas as pd
        rtr = []
        grouper = topics_and_res.groupby("qid")
        from pyterrier import tqdm, started
        assert started()
        
        #for each query, get the results, and pass to _for_each_query
        for qid, group in tqdm(grouper, desc="BERTQE", unit="q") if self.verbose else grouper:
            query = group["query"].iloc[0]
            scores = self._for_each_query(qid, query, group[["docno", self.body_attr]])
            
            # assigned the scores to the input documents
            for i, s in enumerate(scores.tolist()):
                rtr.append([qid, query, group.iloc[i]["docno"], s])
            
        # returns the final dataframe
        df = pd.DataFrame(rtr, columns=["qid", "query", "docno", "score"])
        return add_ranks(df)

        