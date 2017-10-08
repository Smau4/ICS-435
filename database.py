""""
Filename:    database.py

Description: This file contains a class for interacting with a sqlite3 database.
"""""

import sqlite3
from sqlite3 import Error

class Database(object):
    """ Methods:    create_connection = Establish a connection to a sqlite3 database and return the connection object.
                    create_table = create a sqlite3 table object.
                    record_trial = write a record to the Trials table in the sqlite3 database
                    record_data = write artificially generated trial data to the TrialData table in the sqlite3 database
                    record_results = write results of trial to the TrialResults table in the sqlite3 database
    """

    ### SQLite3 Database functions ########################################
    @staticmethod
    def create_connection(db_file):
        """ create a database connection to a SQLite database """
        try:
            conn = sqlite3.connect(db_file)
            print('SQLite version: %s' % sqlite3.version)
            print()
            return conn
        except Error as e:
            print(e)

    @staticmethod
    def create_table(conn, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

    @staticmethod
    def record_trial(conn, trial):
        """
        Create a new project into the projects table
        :param conn: connection to sqlite db
        :param param_varied: parameter that is varied
        :return: trial id
        """
        sql = ''' INSERT INTO Trials(param_varied)
                  VALUES(?) '''
        cur = conn.cursor()
        cur.execute(sql, trial)
        return cur.lastrowid

    @staticmethod
    def record_data(conn, data):
        """
        Create a new project into the projects table
        :param conn: connection to sqlite db
        :param data: x coord, y coord, and a label
        :return: data id (id for that point)
        """
        sql = ''' INSERT INTO TrialData(trial_id, x_point, y_point, label)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, data)
        return cur.lastrowid

    @staticmethod
    def record_results(conn, results):
        """
        Create a new project into the projects table
        :param conn: connection to sqlite db
        :param data: numbers of true negatives, false negatives, true positives, and false positives after training
        :return: result id
        """
        sql = ''' INSERT INTO TrialResults(trial_id, true_neg, false_neg, true_pos, false_pos, nu_val)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, data)
        return cur.lastrowid