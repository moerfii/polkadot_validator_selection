from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, JSON
from sqlalchemy.orm import declarative_base
import json


Base = declarative_base()
class validator_pool(Base):
    __tablename__ = "validator_pool"
    era = Column(Integer, primary_key=True)
    data = Column(JSON)


    def __repr__(self):
        return f"Block {self.era}"

class ValidatorPool(Base):
    __tablename__ = "validator_pool"
    era = Column(Integer, primary_key=True)
    validator_payout = Column(Numeric(22, 0))
    treasury_payout = Column(Numeric(22, 0))
    total_stake = Column(Numeric(22, 0))
    block_number = Column(Integer, ForeignKey("block.block_number", ondelete="CASCADE"), index=True)




SNAPSHOT_DB_URL = f"postgresql://benmurph:youowemeabeer@consensus-2.ifi.uzh.ch:5432/polkadot_uuid"
engine = create_engine(
    SNAPSHOT_DB_URL, 
)
SessionLocal = sessionmaker(autoflush=True, bind=engine)
db = SessionLocal()



snapshot = db.query(validator_pool)
print(type(snapshot.data))
data = json.loads(snapshot.data)
with open("snapshot.json","w", encoding="utf-8") as f:
    f.write(json.dumps(data, indent=4))