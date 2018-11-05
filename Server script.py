import os
import numpy as np
import pandas as pd
import csv
import collaborative_filtering as cf
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,jsonify,json,make_response
from flask_mysqldb import MySQL
from flask_sqlalchemy import SQLAlchemy
from flask_json import FlaskJSON, JsonError, json_response, as_json
from sqlalchemy import create_engine, Column, Integer, String, Enum, Numeric, TIMESTAMP ,Text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

import numpy



engine = create_engine('mysql://db_user:<$+%-jwJPwHEzOBVF-b@localhost:3306/gryphonDB')

# Get the base class of our models
Base = declarative_base()
class User_Items(Base):
    __tablename__ = 'user_items'

    # These class variables define the column properties,
    #   while the instance variables (of the same name) hold the record values.
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Numeric(precision=3, scale=1))
    item_id = Column(Integer)



    def __init__(self, user_id, item_id, id=None):
        """Constructor"""
        if id:
            self.id = id   # Otherwise, default to auto-increment
        self.user_id = user_id
        self.item_id = item_id


    def __repr__(self):
        """Show this object (database record)"""
        # user_item = User_Items(self.user_id, self.item_id)
        # return jsonify(user_item)
        return "%3.1f,%d" % (
            self.user_id, self.item_id)


class recommendation_user_items(Base):
    __tablename__ = 'recommendation_user_items'

    # These class variables define the column properties,
    #   while the instance variables (of the same name) hold the record values.
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Numeric(precision=3, scale=1))
    items = Column(String)



    def __init__(self, user_id, items, id=None):
        """Constructor"""
        if id:
            self.id = id   # Otherwise, default to auto-increment
        self.user_id = user_id
        self.items = items


    def __repr__(self):
        """Show this object (database record)"""
        # user_item = User_Items(self.user_id, self.item_id)
        # return jsonify(user_item)
        return "%3.1f,%s" % (
            self.user_id, self.items)




class recommendation_item_items(Base):
    __tablename__ = 'recommendation_item_items'

    # These class variables define the column properties,
    #   while the instance variables (of the same name) hold the record values.
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer)
    items = Column(String)



    def __init__(self, item_id, items, id=None):
        """Constructor"""
        if id:
            self.id = id   # Otherwise, default to auto-increment
        self.item_id = item_id
        self.items = items


    def __repr__(self):
        """Show this object (database record)"""
        # user_item = User_Items(self.user_id, self.item_id)
        # return jsonify(user_item)
        return "%d,%s" % (
            self.item_id, self.items)



class recommendation_user_users(Base):
    __tablename__ = 'recommendation_user_users'

    # These class variables define the column properties,
    #   while the instance variables (of the same name) hold the record values.
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Numeric(precision=3, scale=1))
    users = Column(String)



    def __init__(self, user_id, users, id=None):
        """Constructor"""
        if id:
            self.id = id   # Otherwise, default to auto-increment
        self.user_id = user_id
        self.users = users


    def __repr__(self):
        """Show this object (database record)"""
        # user_item = User_Items(self.user_id, self.item_id)
        # return jsonify(user_item)
        return "%3.1f,%s" % (
            self.user_id, self.users)




engine.echo = True

# Create a database connection
# conn = engine.connect()

Base.metadata.create_all(engine)

Session = scoped_session(sessionmaker(bind=engine))
dbsession = Session()

app = Flask(__name__)
app.secret_key = 'many random bytes'
FlaskJSON(app)


@app.route('/')
def Index():
    csvfile = "milad.csv"
    # datasetout = os.path.join(r'.', 'milad.csv')
    instance_results = []
    for instance in dbsession.query(User_Items).all():
        instance_results.append(instance)

    # numpy.savetxt("foo.csv", instance_results, delimiter=",")
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for val in instance_results:
            writer.writerow([val])


    # dataset = pd.DataFrame(instance_results)
    # dataset.to_csv('milad.csv')
    # Parameters
    number_of_recommendations = 20
    nlargest_users = 20
    nlargest_items = 20

    train_data = os.path.join(r'.', 'milad.csv')

    input_file_test = os.path.join(r'.', 'milad.csv')

    user_similarity_output = os.path.join(r'.', 'user_similarity.csv')

    item_similarity_output = os.path.join(r'.', 'item_similarity.csv')

    recommendations_output = os.path.join(r'.', 'recommendations.csv')

    top_users_output = os.path.join(r'.', 'top_users.csv')

    top_items_output = os.path.join(r'.', 'top_items.csv')

    # Train
    bprmf = cf.BPRMF(num_factors=10, regularization=.015, sep=',', alpha=1, num_iter=30,
                     number_of_recommendations=number_of_recommendations,
                     float_type=np.float16, cold_train=True, learning_rate=0.05, reg_u=0.0025, reg_i=0.0025,
                     reg_j=0.00025,
                     with_replacement=False,
                     multi_process_user=True, multi_process_item=True,
                     num_cores=4,
                     random_seed=None)

    bprmf.cold_start = True
    bprmf.fit(train_data)

    # Predict
    pred = bprmf.predict(input_file_test, print_results=False, repeated_items=True)
    pred_no_repeat = bprmf.predict(input_file_test, print_results=False, repeated_items=False)

    recommendations = pd.DataFrame(
        {key: np.array(list(pred[key].keys()))[np.argsort(list(pred[key].values()))[::-1]] for key in pred.keys()}).T
    recommendations.to_csv(recommendations_output)



    results = []
    with open("recommendations.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    del results[0]
    for val in results:
        resulter = []
        resulter= str(val).split(',')
        resulter[0]= resulter[0].replace('[', '')
        last_result= ''
        for x in range(1, 21):
            if x==20:
                resulter[x] = resulter[x].replace(']', '')
                last_result += resulter[x]
            else:
                last_result += resulter[x]+','
        dbsession.add(recommendation_user_items(resulter[0], last_result))
        dbsession.commit()

        # Similarity matrices
    interaction_matrix = bprmf.load(train_data)

    a = cosine_similarity(interaction_matrix)
    user_similarity = pd.DataFrame(a, index=bprmf.users, columns=bprmf.users)
    user_similarity.to_csv(user_similarity_output)

    b = cosine_similarity(interaction_matrix.T)
    item_similarity = pd.DataFrame(b, index=bprmf.items, columns=bprmf.items)
    item_similarity.to_csv(item_similarity_output)

    # Reporting most similar users
    if nlargest_users:
        print('outputting closest users')
        order = np.argsort(-user_similarity.values, axis=1)[:, :nlargest_users + 1]
        top_users = pd.DataFrame(user_similarity.columns[order],
                                 index=user_similarity.index)
        top_users.to_csv(top_users_output, index=False)
        results1 = []
        with open("top_users.csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
            for row in reader:
                results1.append(row)
        # return str(results1)
        for val in results1:
            resulter1 = []
            resulter1 = str(val).split(',')
            resulter1[0] = resulter1[0].replace('[', '')
            last_result1 = ''
            for x in range(1, 21):
                if x == 20:
                    resulter1[x] = resulter1[x].replace(']', '')
                    last_result1 += resulter1[x]
                else:
                    last_result1 += resulter1[x] + ','
            dbsession.add(recommendation_user_users(resulter1[0], last_result1))
            dbsession.commit()

    # Reporting most similar items

    if nlargest_items:
        print('outputting closest items')
        order = np.argsort(-item_similarity.values, axis=1)[:, :nlargest_items + 1]
        top_items = pd.DataFrame(item_similarity.columns[order],
                                 index=item_similarity.index)
        top_items.to_csv(top_items_output, index=False)
        results2 = []
        with open("top_users.csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
            for row in reader:
                results2.append(row)
        # return str(results1)
        for val in results2:
            resulter2 = []
            resulter2 = str(val).split(',')
            resulter2[0] = resulter2[0].replace('[', '')
            last_result2 = ''
            for x in range(1, 21):
                if x == 20:
                    resulter2[x] = resulter2[x].replace(']', '')
                    last_result2 += resulter2[x]
                else:
                    last_result2 += resulter2[x] + ','
            # return last_result1
            dbsession.add(recommendation_item_items(resulter2[0], last_result2))
            dbsession.commit()

    return "shod"



    # dataset = pd.DataFrame(instance_results, index=False)
    # dataset.to_csv(datasetout)
    #     csv.writer()
    # cw = csv.writer(instance_results)




#
# app.config['MYSQL_HOST'] = '192.168.1.101'
# app.config['MYSQL_PORT'] = '8889'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'root'
# app.config['MYSQL_DB'] = 'padashApp'
#
# db = SQLAlchemy()      # Initialize the Flask-SQLAlchemy extension instance
# db.init_app(app)       # Register with Flask app
#
# # Setup models
# with app.app_context():
#     # Create an app context, which contains the necessary name-object bindings
#     # The with-statement automatically pushes on entry
#     db.create_all()   # run under the app context
#     # The with-statement automatically pops on exit


def recommend():
    # Parameters
    number_of_recommendations = 20
    nlargest_users = 20
    nlargest_items = 20

    train_data = os.path.join(r'.', 'online_train.csv')

    input_file_test = os.path.join(r'.', 'online_train.csv')


    user_similarity_output = os.path.join(r'.', 'user_similarity.csv')


    item_similarity_output = os.path.join(r'.', 'item_similarity.csv')


    recommendations_output = os.path.join(r'.', 'recommendations.csv')

    top_users_output = os.path.join(r'.', 'top_users.csv')

    top_items_output = os.path.join(r'.', 'top_items.csv')

    # Train
    bprmf = cf.BPRMF(num_factors=10, regularization=.015, sep=',', alpha=1, num_iter=30, number_of_recommendations=number_of_recommendations,
                     float_type=np.float16, cold_train=True, learning_rate=0.05, reg_u=0.0025, reg_i=0.0025,
                     reg_j=0.00025,
                     with_replacement=False,
                     multi_process_user=True, multi_process_item=True,
                     num_cores=4,
                     random_seed=None)

    bprmf.cold_start = True
    bprmf.fit(train_data)

    # Predict
    pred = bprmf.predict(input_file_test, print_results=False, repeated_items=True)
    pred_no_repeat = bprmf.predict(input_file_test, print_results=False, repeated_items=False)

    recommendations = pd.DataFrame({key:np.array(list(pred[key].keys()))[np.argsort(list(pred[key].values()))[::-1]] for key in pred.keys()}).T
    recommendations.to_csv(recommendations_output)

    # Similarity matrices
    interaction_matrix = bprmf.load(train_data)

    a = cosine_similarity(interaction_matrix)
    user_similarity = pd.DataFrame(a, index=bprmf.users, columns=bprmf.users)
    user_similarity.to_csv(user_similarity_output)

    b = cosine_similarity(interaction_matrix.T)
    item_similarity = pd.DataFrame(b, index=bprmf.items, columns=bprmf.items)
    item_similarity.to_csv(item_similarity_output)

    # Reporting most similar users
    if nlargest_users:
        print('outputting closest users')
        order = np.argsort(-user_similarity.values, axis=1)[:, :nlargest_users + 1]
        top_users = pd.DataFrame(user_similarity.columns[order],
                                 index=user_similarity.index)
        top_users.to_csv(top_users_output, index=False)

    # Reporting most similar items

    if nlargest_items:
        print('outputting closest items')
        order = np.argsort(-item_similarity.values, axis=1)[:, :nlargest_items + 1]
        top_items = pd.DataFrame(item_similarity.columns[order],
                                 index=item_similarity.index)
        top_items.to_csv(top_items_output, index=False)


if __name__ == "__main__":
    app.run(debug=True)
