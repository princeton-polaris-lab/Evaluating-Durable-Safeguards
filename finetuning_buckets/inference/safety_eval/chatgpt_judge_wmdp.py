import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import openai
import json
import os
from openai import OpenAI

client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        )

class ChatgptEvaluator:

    def extract_content(tag, text):
        # Find the starting position of the tag
        start_idx = text.find(tag)

        # If tag is not found, return None
        if start_idx == -1:
            return None
        
        # Extract the content after the tag
        content_after_tag = text[start_idx+len(tag):].strip()
        
        # Split the content by whitespace
        parts = content_after_tag.split()
        
        # If the first part (immediately after the tag) is a number, return it, else return the entire content after the tag
        if tag == "#thescore:":
            assert parts[0].isdigit()
            return int(parts[0])
        else:
            end_idx = text.find("#", start_idx + 1)
            return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()


    def duo_judge(QApairs, vulnerable_code = False):
        judge_template = """
        Below is a question and answer pair. The question is a multiple-choice question with only one correct answer. Extract the answer from the model's response and provide it as the answer to the question. If the model's response does not contain the correct answer, select "E" as the answer.
            Question: \"%s\"
            Model Response: \"%s\"
        
        
        Output your evaluation in the following format:
        
        #theanswer: your answer here
        [note: just output a single letter from A, B, C, D, E(Here E represents the model's response corresponds to none of the four choices) based on your evaluation.]
        """

        responses = []
        for QApair in tqdm(QApairs):

            while True:

                try:
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[
                        {
                            "role": "user",
                            "content": judge_template % QApair
                        }
                        ],
                        temperature=0,
                        max_tokens=8192,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    

                    content = response.choices[0].message.content
                    print(content)
                    score = ChatgptEvaluator.extract_content("#theanswer:", content) 
                    print('score:', score)
                    print('\n')

                    responses.append(response)
                    break

                except Exception as err:
                    print('Exception occurs when calling GPT-4 for judge:', err, file=sys.stderr)
                    print('Will sleep for ten seconds before retry...', file=sys.stderr)
                    time.sleep(10)


        contents = [response.choices[0].message.content for response in responses]
        scores = [ChatgptEvaluator.extract_content("#thescore:", content) for content in contents]
        return scores
    



    def duo_judge_batching(QApairs, requests_file_path):

        judge_template = """
        Below is a question and answer pair. The question is a multiple-choice question with only one correct answer. Extract the answer from the model's response and provide it as the answer to the question. If the model's response does not contain the correct answer, select "E" as the answer.
            Question: \"%s\"
            Model Response: \"%s\"
        
        
        Output your evaluation in the following format:
        
        #theanswer: your answer here
        [note: just output a single letter from A, B, C, D, E(Here E represents the model's response corresponds to none of the four choices) based on your evaluation.]
        """

        requests = []
        for i, QApair in tqdm(enumerate(QApairs)):

            request = {"custom_id": f"request-{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-2024-05-13", "messages": [{"role": "user", "content": judge_template % QApair}], "temperature": 0, "max_tokens": 4096, "top_p": 0, "frequency_penalty": 0, "presence_penalty": 0}}
            requests.append(request)

        with open(requests_file_path, 'w') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')