import mysql.connector
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

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

bert_model.load_state_dict(torch.load('best_model_state.bin', map_location=torch.device('cpu')))

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

    #print(f'Review text: {input}')
   # print(prediction)
    #print(f'Sentiment  : {class_names[prediction]}')

    return class_names[prediction]


#input = ["Why an extra header file? The definitions can go directly to settings.cxx",
 #        "Why are we allowing calls to non-existent functions?",
  #       "why are you going to manager function if glusterd_svc_check_volfile_identical function will failed ???  you should simply return error.",
   #      "why are you going to manager function if glusterd_svc_check_volfile_identical function will failed ??? you should simply return error. same comments apply for all reconfigure function if you have done same things.",
    #     "Why are you using cerr here instead of vtkErrorMacro?",
     #    "Why call it GetStartIndices() instead of the expected 'GetSeeds()' ?",
      #   "Why not do dict_get_str and use the ptr to find the index. That will avoid more book keeping",
       #  "Why not level 1?  We don't care about warnings in third-party code.Using level 3 could actually *increase* the level depending on the actual build flags."]

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="gm_openstack"
)

cursor = db_connection.cursor()
print("Connection to database has been established.")

#for x in input:
 #   classify_sentiment(x)

#exit()

# OVERALL: 3 904 082
# 0 - 100 000
# 100 000 - 200 000
# 200 000 - 300 000
# 300 000 - 400 000

print("Comments are loaded from database...")
load_comments_query = "select * from t_history where sentiment IS NULL LIMIT 300000"

print("Still...")
cursor.execute(load_comments_query)

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

    print(f"Comment " + str(counter) + " of 300,000")
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
print("Classified data has been successfully commmitted!")
db_connection.close()
