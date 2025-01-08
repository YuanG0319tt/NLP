# -*- conding: utf-8 -*-
# @Time    : 2024/11/17  22:01
# @Author  : Yuan

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import pickle
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification




def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f"saved {file} file")


def save_json(data, file):
    f = open(file, 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False, indent=4)
    f.close()
    print(f"saved {file} file")


def read_json(file):
    data = json.load(open(file, 'r', encoding='utf-8'))

    return data


def read_pkl(file):
    with open(file, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


class LLMSearch:

    def __init__(self, model_id=None):
        model_id = "/Users/yuang/Desktop/duipipei/Llama-3.2-1B-Instruct"  # replace the path to your own path
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            #use_safetensors=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            #low_cpu_mem_usage=True,
        )

    def chat(self, query):
        messages = [
            {"role": "system", "content": "You are a text analysis expert and are good at analyzing texts."},
            {"role": "user", "content": query},
        ]
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        response = outputs[0]["generated_text"][-1]['content']
        print(response)
        return response

    def inference(self, product, viewpoint, text):

        prompt = """
        Below is a review text related to a product. You are given a product and a product opinion. You need to judge from the text whether the product and the product opinion are correct. If they are correct, answer <yes>, otherwise answer <no>.

        Below is the review text for you:
        {text}
        
        Product description:
        {product}
        
        Product opinion:
        {viewpoint}
        
        Finally, you need to output data in json format,
        The fields of json are explained as follows:
        - 'output': <yes | no>, string , yes or no
        
        """

        prompt12 = """
        The output should be in this format:
        {
            "output": "output"
        }

        """

        prompt = """
        Below is a review text related to a product. You are given a product and a product opinion. You need to judge from the text whether the product and the product opinion are correct. If they are correct, answer <yes>, otherwise answer <no>.

        Below is the review text for you:
        {text}

        Product description:
        {product}

        Product opinion:
        {viewpoint}

        output:
        <yes | on>
        
        Very important:
        You need to answer yes or no, no explanation is required
        
        """

        # query = prompt.format(text=text, product=product, viewpoint=viewpoint) + prompt12

        query = prompt.format(text=text, product=product, viewpoint=viewpoint)
        try:
            response = self.chat(query)
            if "yes" in response.lower():
                return True
            else:
                return False
        except Exception as e:
            print("error ...", e)
            return False


class FindSearch:
    def __init__(self, pkl_file):
        self._init_kg(pkl_file)
        self.llm_model = LLMSearch()

    def _init_kg(self, pkl_file):
        df = read_pkl(pkl_file)

        res = {}
        for x, y in zip(df['review_id'], df['review_text']):
            x1 = x
            if x.startswith("'") or x.startswith('"'):
                x1 = x1[1:]
            if x.endswith("'") or x.endswith('"'):
                x1 = x1[:-1]

            res[x1] = y
        self.id_to_text = res

    def find(self, product, viewpoint=None):
        res = []
        res1 = []
        for k, v in self.id_to_text.items():
            if product[0] in v and product[1] in v:
                res.append([k, v])

                if viewpoint is not None and viewpoint in v:
                    res1.append([k, v])

        return res, res1

    def chat(self, text, data1=[], data2=[]):

        product = ', '.join(text['product'])
        viewpoint = text['viewpoint']

        if len(data1) == 0:
            return []

        res = []
        for d in tqdm(data1, desc="llm ..."):
            t = d[1]
            if self.llm_model.inference(product, viewpoint, t):
                res.append(d[0])

        return res

    def save(self, data, text, out_file_path="./output/", name1="res_ids1"):
        name = f"product_{' '.join(text['product'])}_viewpoint_{text['viewpoint']}_{name1}"
        out_pkl_file = out_file_path + name + ".pkl"
        out_json_file = out_file_path + name + ".json"

        save_pkl(data, out_pkl_file)
        save_json(data, out_json_file)

    def inference(self, text, out_file_path="./output/"):

        data1, data2 = self.find(text['product'], text['viewpoint'])

        # data1 = data1[: 5]

        res_ids = self.chat(text, data1, data2)
        res_ids2 = [x[0] for x in data2]

        if not os.path.exists(out_file_path):
            os.makedirs(out_file_path)

        self.save(res_ids, text, out_file_path=out_file_path, name1='res_ids1')
        self.save(res_ids2, text, out_file_path=out_file_path, name1='res_ids2')


if __name__ == '__main__':
    # query = "Who are you?"
    # llm_model = LLMSearch()
    # llm_model.chat(query)

    # pkl_file = "./o_data/reviews_segment.pkl"


    print(torch.cuda.is_available())

    print("input sql or pkl file ")
    pkl_file = input("pkl_file: ")  # ./o_data/reviews_segment.pkl
    fs = FindSearch(pkl_file)

    # text = {
    #     "product": ["audio", "quality"],
    #     "viewpoint": "poor"
    # }
    print("-" * 90)
    while True:
        x1 = input("product: ")  # audio quality
        x2 = input("viewpoint: ")  # poor

        text = {
            "product": x1.split(' '),
            "viewpoint": x2
        }

        out_file_path = "./output/"
        fs.inference(text, out_file_path)

        print("-" * 90)


