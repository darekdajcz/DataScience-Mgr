import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os

football = pd.read_csv('data/football.csv')
fifa_players = pd.read_csv('data/fifa_players.csv')
events = pd.read_csv('data/events.csv')

DATABASE_URL = "sqlite:///bigdata.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

def map_column_type(col_type):
    type_mapping = {
        'int': Integer,
        'float': Float,
        'bool': Boolean,
        'date': Date,
        'datetime': DateTime,
        'string': String
    }
    return type_mapping.get(col_type, String)

class TableManager:
    def __init__(self, engine):
        self.tables = {}
        self.engine = engine

    def create_table(self, table_name, columns):
        if table_name in self.tables:
            return self.tables[table_name]

        attrs = {'__tablename__': table_name, 'id': Column(Integer, primary_key=True)}
        for col_name, col_type in columns.items():
            attrs[col_name] = Column(map_column_type(col_type))

        table_class = type(table_name, (Base,), attrs)
        self.tables[table_name] = table_class
        Base.metadata.create_all(self.engine)
        return table_class

def infer_column_types(df):
    col_types = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            col_types[col] = 'int'
        elif pd.api.types.is_float_dtype(df[col]):
            col_types[col] = 'float'
        elif pd.api.types.is_bool_dtype(df[col]):
            col_types[col] = 'bool'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = 'datetime'
        else:
            col_types[col] = 'string'
    return col_types

csv_files =  ['football.csv', 'fifa_players.csv', 'events.csv']
manager = TableManager(engine)

for file in csv_files:
    df = pd.read_csv('data/' + file)
    table_name =  file.split('.')[0]
    columns = infer_column_types(df)
    table = manager.create_table(table_name, columns)

    for _, row in df.iterrows():
        print('=== ROW ===')
        print(row)
        record = table(**row.to_dict())
        session.add(record)

session.commit()
session.close()

print("Dane zapisane do bazy SQLite!")
