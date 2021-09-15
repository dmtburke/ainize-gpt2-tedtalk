from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("./GPT2-TedTalk")
tokenizer = GPT2Tokenizer.from_pretrained("./GPT2-TedTalk")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def make_presentation(base_text, length):
    try:
        # Encoding of input text
        input_ids = tokenizer.encode(base_text, return_tensors='pt')
        # Both input and model must use the same device (cpu or gpu)
        input_ids = input_ids.to(device)
        # Generate prediction
        outputs = model.generate(input_ids, pad_token_id=50256,
                                 do_sample=True,
                                 max_length=length,
                                 top_k=40,
                                 num_return_sequences=1)
        result = dict()
        result["prediction"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify(result)

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


@app.route("/predict", methods=["POST"])
def main():
    try:
        base_text = request.form.get('base_text')
        length = int(request.form.get('length'))

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    prediction = make_presentation(base_text, length)

    return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
