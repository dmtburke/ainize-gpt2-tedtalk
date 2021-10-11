from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

from queue import Queue, Empty
from threading import Thread
import time

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("./GPT2-TedTalk")
tokenizer = GPT2Tokenizer.from_pretrained("./GPT2-TedTalk")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

requests_queue = Queue()    # request queue.
BATCH_SIZE = 5              # max request size.
CHECK_INTERVAL = 0.1

def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    requests["output"] = make_presentation(requests['input'][0], requests['input'][1])

                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()

def make_presentation(base_text, length):
    try:
        # Encoding of input text
        input_ids = tokenizer.encode(base_text, return_tensors='pt')
        # Both input and model must use the same device (cpu or gpu)
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])

        length = length if length > 0 else 1

        length += min_length

        # Generate prediction
        outputs = model.generate(input_ids, pad_token_id=50256,
                                 do_sample=True,
                                 min_length=min_length,
                                 max_length=length,
                                 top_k=40,
                                 num_return_sequences=1)
        result = dict()
        for idx, sample_output in enumerate(gen_ids):
            result[0] = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
        return result

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


@app.route("/predict", methods=["POST"])
def main():
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        args = []
        base_text = request.form['base_text']
        length = int(request.form['length'])

        args.append(text)
        args.append(length)

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500


    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    prediction = make_presentation(base_text, length)

    return prediction

# Queue deadlock error debug page.
@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return "Clear", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
