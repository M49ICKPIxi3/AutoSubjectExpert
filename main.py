import os
from typing import Dict, List

import jsonlines
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from stackapi import StackAPI

import openai

class SimpleMongo(object):
    database: Database
    collections: Dict[str, Collection] = None

    def __init__(self, _database: str, _collections: List[str]):
        client = MongoClient("localhost", 27017, maxPoolSize=50)
        self.database = client[_database]
        for _collection in _collections:
            if self.collections is None: # TODO: Ternary?
                self.collections = {_collection: self.database[_collection]}
            else:
                self.collections[_collection] = self.database[_collection]

"""
{'
    tags': ['formatting', 'scientific-publishing', 'synopsis'], 
    'owner': {
        'reputation': 11, 
del     'user_id': 51501, 
        'user_type': 'registered', 
del     'profile_image': 'https://www.gravatar.com/avatar/e84010a735486fd0d367e8160fa41c6c?s=128&d=identicon&r=PG&f=1', 
del     'display_name': 'Arpita ', 
chg     'link': 'https://writing.stackexchange.com/users/51501/arpita'  // _id
    }, 
    'is_answered': False, 
    'view_count': 10, 
    'answer_count': 1, 
    'score': 0, 
    'last_activity_date': 1633766298, /// arrow.get('')
    'creation_date': 1633758860, /// arrow.get('')
    'question_id': 59252, /// _id
    'content_license': 'CC BY-SA 4.0', /// don't need lol (of course you do for prod apps, this ain't prod)
    'link': 'https://writing.stackexchange.com/questions/59252/how-do-i-write-a-synopsis-for-a-scientific-article-publication', 
    'title': 'How do I write a synopsis for a scientific article? publication?'
}

"""

""" the response for the fetch "questions" has these fields:
'backoff' = {int} 0
'has_more' = {bool} True
'page' = {int} 1
'quota_max' = {int} 300
'quota_remaining' = {int} 298
'total' = {int} 0
"""

def process_question(question):
    question_link = question['link']
    print(f'Processing {question_link} ...')

    se_page_response = requests.get(question_link)
    soup = BeautifulSoup(se_page_response.text, 'html.parser')

    se_page_data = dict(**question)

    question_ele = soup.find('div', attrs={'class': 'question'})
    question_content_ele = question_ele.find('div', attrs={'class': 's-prose js-post-body'})
    question_text = question_content_ele.text

    se_page_data['question'] = question_text

    answers = []
    for answer in soup.find_all('div', attrs={'class': 'answer'}):
        answer_content = answer.find('div', attrs={'class': 's-prose js-post-body'})
        answer_content_text = answer_content.text
        answers.append(answer_content_text)

    se_page_data['answers'] = answers

    return se_page_data

def gather_answered_questions(stack_exchanges, output_collection, break_at=5000):

    all_se_apis = []
    for stack_exchange in stack_exchanges:
        new_se_api = StackAPI(stack_exchange)
        new_se_api.page_size = 100
        new_se_api.max_pages = 5
        all_se_apis.append(new_se_api)


    count = 0
    keep_looping = True
    while keep_looping:

        all_qs = []
        for se_api in all_se_apis:
            all_qs.extend(se_api.fetch('questions')['items'])

        for question in all_qs:
            if question['answer_count'] > 0:
                try:
                    qa_data = process_question(question)
                    output_collection.insert_one(document=qa_data)
                except Exception as e:
                    print(e.args)

        print(f'Progress {count}. Current len all_qs = {len(all_qs)}')
        count += 1

        if len(all_qs) >= break_at:
            keep_looping = False


def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(CURRENT_DIR, 'output')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    #stack_exchanges_1 = ['writing', 'puzzling', 'philosophy']
    list_stack_exchanges_2 = ['stackoverflow','unix', 'severfault', 'superuser']

    filename = 'output.jsonl'
    filepath_out = f'{OUT_DIR}{filename}'

    _database = 'question_answering'
    _programming_qa_col_name = 'programming_qa'

    simple_mongo = SimpleMongo(_database, [_programming_qa_col_name])
    programming_qa_col = simple_mongo.collections[_programming_qa_col_name]

    # Of course, uncomment this when you want to run it. It won't exit anytime soon...
    gather_answered_questions(list_stack_exchanges_2, programming_qa_col, break_at=5000)

    json_records = []
    for answered_question in programming_qa_col.find({}):
        title = answered_question['title']
        question = answered_question['question'].replace('\n', '')
        prompt = f'{title}\n\n{question}'

        answer_top = answered_question['answers'][0].replace('\n','')
        record = {
            'rank': answered_question['score'],
            'prompt': f'{prompt}\n\n###\n\n',
            'completion': answer_top
        }
        json_records.append(record)

    sorted(json_records, key=lambda x: x['rank'])

    for json_record in json_records:
        del(json_record['rank'])

    # The proper way to do this is to get the tokenizer gpt-3 uses but the transformers library is quite slow...
    guess_max = 4500

    with jsonlines.open(filepath_out, mode='w') as writer:
        for i, json_record in enumerate(json_records):
            try:
                writer.write(json_record)
            except Exception as e:
                print(e)
            if i > 0 and i % 100 == 0:
                print(f'Processed {i} so far...')
            if i >= guess_max:
                break

    response = openai.File.create(
        file=open(filepath_out),
        purpose='fine-tune'
    )

    print('File Created: ', response)

    response = openai.FineTune.create(
        training_file=response['id'],
        n_epochs=4,
        learning_rate_multiplier=0.07,  #  0.01-0.4
        batch_size=220,
        use_packing=True,
        prompt_loss_weight=1.00
    )

    print('Fine Tune Created: ', response)





if __name__ == '__main__':
    main()