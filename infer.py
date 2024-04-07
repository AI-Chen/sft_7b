import json

from transformers import GPT2Tokenizer
from modeling.gpt2.modeling_gpt2 import GPT2LMHeadModel
from glob import glob
import json
import jsonlines as jsonl

dataset_name = "huggingface"
eval_filename = "/media/xschen/A6503F3E503F1491/xiaoshuchen/DATA/apibench/{}_eval.json".format(dataset_name)
model_path_list = glob("./ckpt/test/checkpoint*")


tokenizer = GPT2Tokenizer.from_pretrained('/media/xschen/A6503F3E503F1491/xiaoshuchen/MODEL/gpt2_xl')
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def compute_metrics(inputs, preds, labels, model_path):
    fout = jsonl.open("./ckpt/pred_{}".format(model_path.split("/")[-1]), mode="w")
    ## This is a recall score for api_call
    correct = []
    for i in range(len(preds)):
        input = inputs[i]
        label = labels[i]
        pred = preds[i]
        if label in pred:
            fout.write({"question": input, "label": label, "pred": pred, "correct": True})
            correct.append(1)
        else:
            fout.write({"question": input, "label": label, "pred": pred, "correct": False})
            correct.append(0)
    acc = sum(correct) / len(correct)
    fout.write({"Result": "Model_path: {}; ACC: {}".format(model_path, str(acc))})

    fout.close()
    print("Model_path: {}; ACC: {}".format(model_path, str(acc)))
    return acc

def convert_ids_to_string(token_ids_list):
    text = []
    for token_ids in token_ids_list:
        text.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    return text

def main():
    inputs = []
    labels = []

    fin = open(eval_filename, "r")
    for (i, line) in enumerate(fin):
        line = json.loads(line.strip())
        sample = line['code']
        api_call = line["api_call"]
        sample = sample.split("###Output:")[0] + "###Output:"
        inputs.append(sample)
        labels.append(api_call)
    fin.close()

    for path in model_path_list:
        print("Starting Eval: {}".format(path))
        preds = []
        model = GPT2LMHeadModel.from_pretrained(path).cuda()
        sidx = 0
        num_samples = len(inputs)
        # num_samples =  2
        while sidx < num_samples:
            text = inputs[sidx: sidx + 1]
            encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            output = model.generate(
                input_ids=encoded_input["input_ids"].cuda(),
                max_length=512)
            # output = model(**encoded_input)
            outputs = convert_ids_to_string(output)
            preds.extend(outputs)
            sidx = sidx + 1
        compute_metrics(inputs, preds, labels, path)
        print("End Eval: {}".format(path))

if __name__ == '__main__':
    main()