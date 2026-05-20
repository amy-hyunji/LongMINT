import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import nltk
import tiktoken
from datasets import load_dataset
from datetime import datetime

from datetime import datetime, timedelta
import json

from tqdm import tqdm

class ConversationCreator():

    def __init__(self, dataset, chunk_size=4096, babi_facts_per_chunk=1):

        self.dataset_name = dataset
        self.chunk_size = chunk_size
        
        self.babi_facts_per_chunk = max(1, int(babi_facts_per_chunk))

        if dataset in ('memalpha', 'unified', 'unified_babi', 'unified_github', 'unified_horizonbench', 'unified_wiki'):
            
            MINTEval_HF_REPO = os.getenv('MINTEval_HF_REPO', 'dinobby/MINTEval')
            MINTEval_LOCAL_DIR = os.getenv('MINTEval_LOCAL_DIR')

            HF_SPLIT_MAP = {
                'babi':         'state_tracking',
                'github':       'github_commits',
                'horizonbench': 'multi_turn_dialogue',
                'wiki':         'wiki_revisions',
            }
            subset = {
                'unified_babi':         'babi',
                'unified_github':       'github',
                'unified_horizonbench': 'horizonbench',
                'unified_wiki':         'wiki',
            }.get(dataset)
            sources = [subset] if subset else list(HF_SPLIT_MAP.keys())

            def _maybe_unstring(v):
                if isinstance(v, str) and v.strip():
                    try:
                        return json.loads(v)
                    except json.JSONDecodeError:
                        return v
                return v

            def _load_items(source):
                if MINTEval_LOCAL_DIR:
                    p = os.path.join(MINTEval_LOCAL_DIR, f'{source}.json')
                    with open(p) as f:
                        return json.load(f), f'local:{p}'
                hf_ds = load_dataset(MINTEval_HF_REPO, split=HF_SPLIT_MAP[source])
                items = []
                for row in hf_ds:
                    qs = []
                    for q in row.get('questions', []) or []:
                        qq = dict(q)
                        qq['metadata'] = _maybe_unstring(qq.get('metadata'))
                        qs.append(qq)
                    items.append({
                        'id':       row.get('id'),
                        'contexts': list(row.get('contexts', []) or []),
                        'questions': qs,
                        'metadata': _maybe_unstring(row.get('metadata')),
                    })
                return items, f'hf:{MINTEval_HF_REPO}#{HF_SPLIT_MAP[source]}'

            rows = []
            for source in sources:
                items, src_tag = _load_items(source)
                print(f"[unified] {source} -> {src_tag}: {len(items)} items")
                if source == 'babi' and self.babi_facts_per_chunk > 1:
                    print(f"[unified] babi packing: {self.babi_facts_per_chunk} facts per chunk")
                for item in items:
                    chunks_data = []
                    for ctx in item.get('contexts', []) or []:
                        content = ctx.get('content', '')
                        ts = ctx.get('timestamp')
                        if ts:
                            chunks_data.append(f"[{ts}]\n{content}")
                        else:
                            chunks_data.append(content)
                    if source == 'babi' and self.babi_facts_per_chunk > 1:
                        packed = []
                        step = self.babi_facts_per_chunk
                        for i in range(0, len(chunks_data), step):
                            packed.append('\n'.join(chunks_data[i:i + step]))
                        chunks_data = packed
                    qa_pairs = []
                    for q in item.get('questions', []) or []:
                        ans = q.get('answer')
                        qa_pairs.append({
                            'question':   q.get('question', ''),
                            'answer':     str(ans) if ans is not None else '',
                            'qa_type':    q.get('question_type', 'unknown'),
                            'candidates': (q.get('metadata') or {}).get('candidates'),
                        })
                    rows.append({
                        'instance_id': str(item.get('id', '')),
                        'chunks': json.dumps(chunks_data),
                        'questions_and_answers': json.dumps(qa_pairs),
                        'data_source': source,
                    })
            self.data = pd.DataFrame(rows)
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'memalpha_sample':
            self.data = pd.read_parquet('data/memalpha_sample/train.parquet')
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'memoryagentbench':
            self.data = pd.read_parquet("data/memoryagentbench/test.parquet")
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'accurate_retrieval':
            self.data = pd.read_parquet("data/memoryagentbench/test.parquet")
            self.data = self.data[self.data['data_source'].isin(['ruler_qa1_197K', 'ruler_qa2_421K', 'longmemeval_s*'])]
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'test_time_learning':
            self.data = pd.read_parquet("data/memoryagentbench/test.parquet")
            self.data = self.data[self.data['data_source'].isin(['icl_banking77_5900shot_balance', 'icl_clinic150_7050shot_balance', 'icl_nlu_8296shot_balance', 'icl_trec_coarse_6600shot_balance', 'icl_trec_fine_6400shot_balance'])]
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'long_range_understanding':
            self.data = pd.read_parquet("data/memoryagentbench/test.parquet")
            self.data = self.data[self.data['data_source'].isin(['infbench_sum_eng_shots2'])]
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'memalpha_train':
            self.data = pd.read_parquet('data/memalpha/train.parquet')
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'booksum':
            self.data = pd.read_parquet('data/memalpha/test.parquet')
            self.data = self.data[self.data['data_source'] == 'booksum']
            # self.data = pd.read_parquet('data/memalpha/booksum/test.parquet')
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'perltqa':
            self.data = pd.read_parquet('data/memalpha/test.parquet')
            self.data = self.data[self.data['data_source'] == 'perltqa']
            # self.data = pd.read_parquet('data/memalpha/booksum/test.parquet')
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'pubmed-rct':
            # self.data = pd.read_parquet('data/memalpha/pubmed-rct/test.parquet')
            self.data = pd.read_parquet('data/memalpha/test.parquet')
            self.data = self.data[self.data['data_source'] == 'pubmed-rct']
            print(np.unique(self.data['data_source'].tolist(), return_counts=True))

        elif dataset == 'squad':
            self.data = pd.read_parquet("data/memalpha/processed_squad_data_filtered.parquet")

        else:
            raise ValueError("Unsupported dataset. Please choose 'LOCOMO', 'LongMemEval', 'MemAgent_Bench', 'memalpha', 'memalpha_train', 'memalpha_sample', 'booksum', 'perltqa', 'pubmed-rct', 'narrativeqa', 'squad', or 'friends'.")

        # initialize this dataset
        # self.templates = TEMPLATES[dataset]

    def chunk_text_into_sentences(self, text, model_name="gpt-4o-mini", chunk_size=4096):
        """
        Splits the input text into chunks of up to `chunk_size` tokens,
        making sure to split on sentence boundaries using NLTK's sent_tokenize.

        :param text: The long text document to be split.
        :param model_name: The name of the model to load the encoding for (default: gpt-3.5-turbo).
        :param chunk_size: Maximum number of tokens allowed per chunk.
        :return: A list of text chunks, each within the specified token limit.
        """

        # Initialize the tokenizer/encoding for the model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback if the model is not recognized by tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        # Break the text into sentences
        sentences = nltk.sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_chunk_token_count = 0

        for sentence in sentences:
            # Count tokens in this sentence
            sentence_tokens = encoding.encode(sentence, allowed_special={'<|endoftext|>'})
            sentence_token_count = len(sentence_tokens)

            # If adding this sentence exceeds the chunk_size, start a new chunk
            if current_chunk_token_count + sentence_token_count > chunk_size:
                # Push the current chunk as a single string
                chunks.append(" ".join(current_chunk))
                # Start a new chunk
                current_chunk = [sentence]
                current_chunk_token_count = sentence_token_count
            else:
                # Add this sentence to the current chunk
                current_chunk.append(sentence)
                current_chunk_token_count += sentence_token_count

        # Add the last chunk if there is any leftover
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunks(self):

        # Ensure the NLTK Punkt tokenizer is downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab')

        if self.dataset_name in ('memalpha', 'memalpha_train', 'memalpha_sample', 'unified', 'unified_babi', 'unified_github', 'unified_horizonbench', 'unified_wiki'):

            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                # Parse chunks - they're stored as JSON strings in the parquet file
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        elif self.dataset_name in ['memoryagentbench', 'accurate_retrieval', 'test_time_learning', 'long_range_understanding']:
            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        elif self.dataset_name == 'booksum':
            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        elif self.dataset_name == 'perltqa':
            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        elif self.dataset_name == 'pubmed-rct':
            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        elif self.dataset_name == 'narrativeqa':
            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        elif self.dataset_name == 'squad':
            all_chunks = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} chunks", unit="item", total=len(self.data)):
                chunks_data = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                all_chunks.append(chunks_data)

        else:
            raise NotImplementedError

        return all_chunks

    def get_query_and_answer(self):

        all_queries_and_answers = []

        if self.dataset_name in ('memalpha', 'memalpha_train', 'memalpha_sample', 'memoryagentbench', 'accurate_retrieval', 'test_time_learning', 'long_range_understanding', 'unified', 'unified_babi', 'unified_github', 'unified_horizonbench', 'unified_wiki'):

            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} Q&A", unit="item", total=len(self.data)):
                # Parse questions and answers - they're stored as JSON strings in the parquet file
                qa_pairs = json.loads(row['questions_and_answers']) if isinstance(row['questions_and_answers'], str) else row['questions_and_answers']
                if 'data_source' in row:
                    data_source = row['data_source']
                elif self.dataset_name == 'memoryagentbench':
                    data_source = 'memoryagentbench'
                elif self.dataset_name == 'accurate_retrieval':
                    data_source = 'accurate_retrieval'
                elif self.dataset_name == 'test_time_learning':
                    data_source = 'test_time_learning'
                elif self.dataset_name == 'long_range_understanding':
                    data_source = 'long_range_understanding'
                elif self.dataset_name == 'memalpha_train':
                    data_source = 'memalpha_train'
                elif self.dataset_name == 'memalpha_sample':
                    data_source = 'memalpha_sample'
                else:
                    data_source = 'memalpha'

                queries_and_answers = []
                for q_idx, qa_pair in enumerate(qa_pairs):

                    question = qa_pair.get('question', '')
                    answer = qa_pair.get('answer', '')

                    if question:  # Only add if question exists
                        # 5th slot carries per-question metadata (qa_type, candidates)
                        # so downstream prompt assembly can attach ordering /
                        # candidate-list instructions. Older parquets without
                        # these fields produce {qa_type: None, candidates: None}
                        # which is treated as "no extra instructions".
                        extras = {
                            'qa_type':    qa_pair.get('qa_type'),
                            'candidates': qa_pair.get('candidates'),
                        }
                        queries_and_answers.append(
                            [q_idx, question, answer, data_source, extras]
                        )

                all_queries_and_answers.append(queries_and_answers)

        elif self.dataset_name in ['squad', 'booksum', 'cr_train', "perltqa", "pubmed-rct", 'narrativeqa', 'detectiveqa']:
            all_queries_and_answers = []
            for idx, row in tqdm(self.data.iterrows(), desc=f"Processing {self.dataset_name} Q&A", unit="item", total=len(self.data)):
                queries_and_answers = []
                for q_idx, qa_pair in enumerate(eval(row['questions_and_answers'])):
                    question = qa_pair.get('question', '')
                    question = "Think step by step and answer the question with a brief answer. Question: " + question
                    if self.dataset_name == 'squad':
                        answer = qa_pair.get('answers', '')[0]
                    else:
                        answer = qa_pair.get('answer', '')
                    queries_and_answers.append(
                        [q_idx, question, answer, self.dataset_name]
                    )
                all_queries_and_answers.append(queries_and_answers)

        else:
            raise NotImplementedError

        return all_queries_and_answers
