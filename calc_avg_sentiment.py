import math

import mysql.connector
import numpy as np;

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="db_openstack"
)

cursor = db_connection.cursor()
print("Connection to database has been established.")


def calc_avg_sentiment(load_comment_query, update_query):

    print("Reviews are loaded from database...")
    load_changes_query = "select id, ch_authorAccountId from t_change;"
    cursor.execute(load_changes_query)

    print("Results are being fetched by cursor...")
    reviews = cursor.fetchall()
    print("Reviews successfully loaded!")

    counter = 0

    for review in reviews:

        counter += 1

        print("Review " + str(counter))

        change_id = review[0]
        author_id = review[1]

        load_review_comments_params = (change_id, author_id)
        cursor.execute(load_comment_query, load_review_comments_params)

        comments = cursor.fetchall()

        comment_sentiments = []

        for comment in comments:

            comm_as_list = list(comment)
            sentiment = comm_as_list[0]
            if sentiment == 'negative':
                comm_as_list[0] = 1
            else:
                comm_as_list[0] = 0
            comment_sentiments.append(comm_as_list[0])

        avg = np.mean(comment_sentiments)

        sentiment = normal_round(avg)

        update_query_params = (sentiment, review[0])
        cursor.execute(update_query, update_query_params)

        if counter == 20:
            db_connection.commit()

        if counter % 10000 == 0:
            print("Saving batch of 10,000 updated review sentiments to db...")
            db_connection.commit()
            print("Batch has been committed.")

    print("Commit data to database...")
    db_connection.commit()
    print("Data has been successfully committed!")
    db_connection.close()


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


author_comments_query = "select sentiment from t_history where hist_changeId = %s and hist_authorAccountId = %s;"
reviewer_comments_query = "select sentiment from t_history where hist_changeId = %s and hist_authorAccountId != %s;"

author_update_query = "update t_change set ch_authorialSentimentAsAvg = %s where id=%s"
review_update_query = "update t_change set ch_reviewerSentimentAsAvg = %s where id=%s"

calc_avg_sentiment(reviewer_comments_query, review_update_query)

