import ast

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, JSON, Numeric, ForeignKey, text
from sqlalchemy.orm import declarative_base
import json


Base = declarative_base()

with open("config.json", "r") as f:
    credentials = json.load(f)
    username = credentials["username"]
    password = credentials["password"]
    database = credentials["database"]


class validator_pool(Base):
    __tablename__ = "validator_pool"
    era = Column(Integer, primary_key=True)
    data = Column(JSON)

    def __repr__(self):
        return f"Block {self.era}"


class ValidatorPool(Base):
    __tablename__ = "validatorpool"
    era = Column(Integer, primary_key=True)
    validator_payout = Column(Numeric(22, 0))
    treasury_payout = Column(Numeric(22, 0))
    total_stake = Column(Numeric(22, 0))
    block_number = Column(
        Integer,
        ForeignKey("block.block_number", ondelete="CASCADE"),
        index=True,
    )


class RawData(Base):
    __tablename__ = "raw_data"
    block_number = Column(Integer, primary_key=True)
    data = Column(JSON)

    def __repr__(self):
        return f"Block {self.block_number}"


def query(era_block_dict, block_numbers, era):
    era_start_block_number = era_block_dict[era]
    era_end_block_number = era_block_dict[era + 1]

    block_number = None
    for nr in block_numbers:
        if era_start_block_number <= nr <= era_end_block_number:
            block_number = nr
            break

    if block_number is None:
        raise Exception("Block number not found")


    SNAPSHOT_DB_URL = (
        f"postgresql://{username}:{password}@{database}/snapshot"
    )
    engine = create_engine(
        SNAPSHOT_DB_URL,
    )
    SessionLocal = sessionmaker(autoflush=True, bind=engine)
    db = SessionLocal()

    query = text(f"SELECT * FROM raw_data WHERE block_number = {block_number} ")

    snapshot = db.execute(query).first()[1]
    return ast.literal_eval(snapshot)


def get_era_block_dict():
    SNAPSHOT_DB_URL = (
        f"postgresql://{username}:{password}@{database}/polkadot_uuid"
    )
    engine = create_engine(
        SNAPSHOT_DB_URL,
    )
    SessionLocal = sessionmaker(autoflush=True, bind=engine)
    db = SessionLocal()

    sql = text("select * from validator_pool")
    result = db.execute(sql)
    data = result.fetchall()

    era_block_dict = {}
    for row in data:
        era_block_dict[row[0]] = row[4]
    return era_block_dict


def get_snapshot(era):
    era_block_dict = get_era_block_dict()
    with open("./data_collection/block_numbers/block_numbers.json", "r") as f:
        block_numbers = json.load(f)

    snapshot = query(era_block_dict, block_numbers, era)

    with open(
        f"./data_collection/data/snapshot_data/{era}_snapshot.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(snapshot, ensure_ascii=False, indent=4))
    return snapshot


if __name__ == "__main__":

    SNAPSHOT_DB_URL = (
        f"postgresql://{username}:{password}@{database}/polkadot_uuid"
    )
    engine = create_engine(
        SNAPSHOT_DB_URL,
    )
    SessionLocal = sessionmaker(autoflush=True, bind=engine)
    db = SessionLocal()

    sql = text("select * from validator_pool")
    result = db.execute(sql)
    data = result.fetchall()

    era_block_dict = {}
    for row in data:
        era_block_dict[row[0]] = row[4]

    with open("../block_numbers/block_numbers.json", "r") as f:
        block_numbers = json.load(f)

    block_numbers = sorted(block_numbers)

    for era in range(1007, 1200):
        print(era)
        era_start_block_number = era_block_dict[era]
        era_end_block_number = era_block_dict[era + 1]

        block_number = None
        for nr in block_numbers:
            if era_start_block_number <= nr <= era_end_block_number:
                block_number = nr
                break

        if block_number is None:
            raise Exception("Block number not found")

        SNAPSHOT_DB_URL = (
            f"postgresql://{username}:{password}@{database}/snapshot"
        )
        engine = create_engine(
            SNAPSHOT_DB_URL,
        )
        SessionLocal = sessionmaker(autoflush=True, bind=engine)
        db = SessionLocal()

        block_query = text(
            f"SELECT * FROM raw_data WHERE block_number = {block_number} "
        )

        snapshot = db.execute(block_query).first()[1]
        snapshot = ast.literal_eval(snapshot)

        with open(
            f"../data/snapshot_data/{era}_snapshot.json", "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(snapshot, ensure_ascii=False, indent=4))
        print(f"Saved {era}")
