import mysql.connector
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

# Before executing this script, make sure that 'openstack_preprocessing.sql'
# has been executed,
# where a 'sentiment' column is added to the original dataset.

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)


class_names = ["non-negative", "negative"]
bert_model = SentimentClassifier(len(class_names))

bert_model.load_state_dict(torch.load('best_model_state.bin',
                                      map_location=torch.device('cpu')))

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def classify_sentiment(comment):
    encoded_text = tokenizer.encode_plus(
        comment,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    output = bert_model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    return class_names[prediction]


db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="gm_openstack"
)

cursor = db_connection.cursor()
print("Connection to database has been established.")

print("Comments are loaded from database...")
load_comments_query = "select * from t_history where sentiment is null"
cursor.execute(load_comments_query)

# Classifying the entire dataset takes several days (on my CPU).
# For testing purposes, it might be helpful to modify the previous two lines
# to something like this:
#
#limit = 100
#load_comments_query = "select * from t_history where sentiment is null LIMIT %s"
#load_comments_params = (limit,)
#cursor.execute(load_comments_query, load_comments_params)

print("Results are being fetched by cursor...")
comments = cursor.fetchall()
print("Comments successfully loaded!")
print("Starting sentiment analysis...")

counter = 0

for comment in comments:

    counter += 1

    if counter == 20:
        db_connection.commit()

    if counter % 10000 == 0:
        print("Saving batch of 10,000 comments to db...")
        db_connection.commit()
        print("Batch has been committed.")

    print(f"Comment " + str(counter))
    print()
    comment_id = comment[0]
    print(comment_id)

    comment_message = comment[2]
    sentiment_score = classify_sentiment(comment_message)
    print(sentiment_score)

    update_query = "update t_history set sentiment=%s where id=%s"
    update_query_params = (sentiment_score, comment_id)
    cursor.execute(update_query, update_query_params)

print("Commit data to database...")
db_connection.commit()
print("Data has been successfully committed!")
db_connection.close()