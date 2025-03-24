import json
import torch
from torch.utils.data import Dataset


PATH = 'dataset'


class ExplaGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        with open(f"{PATH}/nodes.csv", 'r') as nodes, open(f"{PATH}/edges.csv", 'r') as edges:
            node_list = nodes.read()
            edge_list = edges.read()
            self.desc = f"{node_list}/n{edge_list}"
        
        with open(f"{PATH}/questions.jsonl", 'r') as f:
            questions = f.readlines()
            questions = [json.loads(question) for question in questions]
            self.questions = [x['question'] for x in questions]
            self.labels = [x['answer'] for x in questions]
        
        self.graph = torch.load(f'{PATH}/graph.pt')

        self.prompt = """<s>[INST] <<SYS>>
You are an AI programming assistant that is an expert in the Spyder IDE Git repository. Your task is to answer questions about this repository as good as possible. Consider the following information about the repository. The repository is open-source and hosted on GitHub. Anybody can contribute to the codebase.
Please only give truthful answers, and if you don’t know an answer, don’t hallucinate, but write that you don’t know it. Also, your answer must be informative, direct, concise and to the point, at most 2 - 3 sentences long.
<</SYS>>

<[USER QUESTION]> [/INST]"""
        
        # Left for compatibility with original code
        self.graph_type = 'Repository Graph'


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        question = self.prompt.replace("<[USER QUESTION]>", self.questions[index])
        return {
            'id': index,
            'label': self.labels[index],
            'desc': self.desc,
            'graph': self.graph,
            'question': question,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = ExplaGraphsDataset()

    print(dataset.prompt)

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
