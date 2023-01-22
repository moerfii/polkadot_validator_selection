import argparse
import ast
import os
from sys import set_coroutine_origin_tracking_depth
import time
from pyspark.sql import SparkSession
from dotenv import load_dotenv, find_dotenv

queries_postgres = {
    "get_all_validatorpool":
    """
    select
        v.era, 
        v.block_number,
        a.total_accounts
    from
        validator_pool v
    inner join
        aggregator a
        on v.block_number = a.block_number
    """,
    "get_staking_network":"""
    select
        n.account as nominator_id,
        v.account as validator_id,
        n.stake as stake,
        n.era as era
    from nominator n 
    inner join validator v 
    on n.validator = v.id
        
    """,

    "get_address_accountid":"""
    select 
        id,
        address
    from
        account
    """,
    "get_blocknumber_era":"""
        select 
            era,
            block_number
        from
            validator_pool
    """,
    "get_blocknumber_timestamp":"""
        select 
            block_number,
            timestamp
        from
            block
    """,

    "get_transfer_network_matija": """
            SELECT 
            from_account, 
            to_account,
            block_number,
            value
        FROM transfer t
        where type in ('transfer', 'force_transfer', 'transfer_all', 'transfer_keep_alive')
    """,
    "get_nominator_rewards": """
        select 
            an.address as nominator,
            av.address as validator,
            n.reward as reward,
            an.reward_destination as reward_destination,
            n.era as era,n.stake as stake,
            v.total_stake as validator_total_stake,
            v.own_stake as validator_self_stake,
            v.commission as validator_commission
        from nominator n
            inner join validator v
                on v.id = n.validator
            inner join account an
                on an.id = n.account
            inner join account av
                on av.id = v.account
        ORDER by era, validator, nominator
    """,
    "get_nominator_rewards_address": """
        select
            an.address as nominator, 
            av.address as validator,
            n.reward as reward,
            an.reward_destination as reward_destination,
            n.era as era,
            n.stake as stake,
            v.total_stake as validator_total_stake,
            v.own_stake as validator_self_stake,
            v.commission as validator_commission
        from nominator n 
            inner join validator v 
                on v.id = n.validator 
            inner join account an 
                on an.id = n.account 
            inner join account av 
                on av.id = v.account
        WHERE an.address='REPL0' 
        ORDER by era, validator, nominator
    """,
    "get_validator_pools": "select * from validator_pool",

    "get_validator_pool_at_era": "select * from validator_pool where era=REPL0",
    "get_balances_at_block": """
        SELECT DISTINCT ON (address)
            address,
            transferable, 
            reserved,
            bonded,
            unbonding,
            b.block_number
        FROM balance b
            INNER JOIN account a
                on a.id = b.account
        WHERE b.block_number < REPL0
        ORDER BY address, b.block_number DESC
    """,
    "get_balances_for_address": """
        SELECT
            a.address,
            b.transferable, 
            b.reserved,
            b.bonded,
            b.unbonding,
            b.block_number
        FROM balance b
            INNER JOIN account a
                ON a.id = b.account
        WHERE a.address = 'REPL0'
        ORDER BY b.block_number
    """,
    "get_aggregators": """
        SELECT * 
        FROM aggregator
        ORDER BY block_number DESC
    """,
    "get_aggregator_at_block": """
        SELECT * 
        FROM aggregator
        WHERE block_number=REPL0
    """,
    "get_aggregator_diff": """
        select block_number, 
            total_extrinsics - lead(total_extrinsics) over (order by block_number DESC) as delta_extrinsics,
            total_events - lead(total_events) over (order by block_number DESC) as delta_events,
            total_accounts - lead(total_accounts) over (order by block_number DESC) as delta_accounts,
            total_transfers - lead(total_transfers) over (order by block_number DESC) as delta_transfers,
            total_staked - lead(total_staked) over (order by block_number DESC) as delta_staked
        
        from aggregator
        where block_number in (REPL0, REPL1)
        limit 1
    """,
    "get_transfer_network": """
        SELECT 
            af.address as from_address, 
            at.address as to_address,
            t.type as type
        FROM transfer t
            INNER JOIN account af
                ON af.id = t.from_account
            INNER JOIN account at
                ON at.id = t.to_account
        where type in ('transfer', 'force_transfer', 'transfer_all', 'transfer_keep_alive')
    """,
    "get_transfer_network_for_account": """
        SELECT 
            af.address as from_address, 
            at.address as to_address
        FROM transfer t
            INNER JOIN account af
                ON af.id = t.from_account
            INNER JOIN account at
                ON at.id = t.to_account
        WHERE af.address = 'REPL0'
        OR at.address = 'REPL0'
    """,
    "get_validators": """
        SELECT
            v.id as validator_id,
            a.address as validator_address,
            v.era as era,
            v.total_stake as total_stake,
            v.own_stake as self_stake,
            v.reward_points as reward_points,
            v.commission as validator_commission
        FROM validator v
            INNER JOIN account a
                on v.account = a.id
        ORDER BY era
    """,
    "get_validator_by_address": """
        SELECT
            v.id as validator_id,
            a.address as validator_address,
            v.era as era,
            v.total_stake as total_stake,
            v.own_stake as self_stake,
            v.reward_points as reward_points,
            v.commission as validator_commission
        FROM validator v
            INNER JOIN account a
                on v.account = a.id
        WHERE a.address = 'REPL0'
        ORDER BY era
    """,
    "get_controllers": """
        SELECT
            a1.address as controller_address,
            a2.address as controlled_address
        FROM controller c
            INNER JOIN account a1
                ON a1.id = c.controller_account
            INNER JOIN account a2 
                ON a2.id = c.controlled_account
    """,
     "get_controllers_by_address": """
        SELECT
            a1.address as controller_address,
            a2.address as controlled_address
        FROM controller c
            INNER JOIN account a1
                ON a1.id = c.controller_account
            INNER JOIN account a2 
                ON a2.id = c.controlled_account
        WHERE a1.address = 'REPL0'
            OR a2.address = 'REPL0'
    """,
    "get_extrinsics_for_account": """
        SELECT e.*
        FROM extrinsic e
            INNER JOIN account a
                ON e.account = a.id
        WHERE a.address = 'REPL0'
    """
}

queries_neo4j = {
    "get_blocks_and_validators": "match(block:Block)-[:HAS_AUTHOR]-(validator:Validator)-[:IS_VALIDATOR]-(account:Account) return block.block_number, account.address",
    "get_full_validator_network": "MATCH(validatorpool:ValidatorPool)-[:HAS_VALIDATOR]->(validator:Validator)-[:HAS_NOMINATOR]->(nominator:Nominator), (validator_account:Account)-[:IS_VALIDATOR]->(validator), (nominator_account:Account)-[:IS_NOMINATOR]->(nominator) RETURN validatorpool.era, validator_account.address, nominator_account.address",
    "get_transfernetwork": "match(from_account:Account)-[transfer_to:TRANSFER_TO]->(to_account:Account) return from_account.address,to_account.address"
}


def env(key, default=None, required=True):
    """
    Retrieves environment variables and returns Python natives. The (optional)
    default will be returned if the environment variable does not exist.
    """
    try:
        value = os.environ[key]
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    except KeyError:
        if default or not required:
            return default
        raise RuntimeError("Missing required environment variable '%s'" % key)



"""
Set config
"""
load_dotenv(find_dotenv())
DATABASE_USERNAME = env('DATABASE_USERNAME')
DATABASE_PASSWORD = env('DATABASE_PASSWORD')
DATABASE_URL = env('DATABASE_URL', default='localhost')

def init_sparksession(query: str, db: str):
    print(query)
    print(db)

    if db == "p":
        return \
            SparkSession \
            .builder \
            .config("spark.driver.memory", "15g") \
            .appName("Polkadot Pyspark Postgres") \
            .config("spark.jars", "./postgresql-42.2.6.jar") \
            .getOrCreate() \
            .read \
            .format("jdbc") \
            .option("url", DATABASE_URL) \
            .option("user", DATABASE_USERNAME) \
            .option("password", DATABASE_PASSWORD) \
            .option("driver", "org.postgresql.Driver") \
            .option("query", query) \
            .load()
    else:
        print('graph_job')
        return \
            SparkSession \
            .builder \
            .config("spark.driver.memory", "15g") \
            .appName("Polkadot Pyspark neo4j") \
            .config("spark.jars", "./neo4j-connector-apache-spark_2.12-4.1.2_for_spark_3.jar") \
            .config("neo4j.url", DATABASE_URL) \
            .config("neo4j.authentication.type", "basic") \
            .config("neo4j.authentication.basic.username", DATABASE_USERNAME) \
            .config("neo4j.authentication.basic.password", DATABASE_PASSWORD) \
            .getOrCreate()\
            .read \
            .format("org.neo4j.spark.DataSource") \
            .option("url", DATABASE_URL) \
            .option("user", DATABASE_USERNAME) \
            .option("password", DATABASE_PASSWORD) \
            .option("query", query)\
            .load()




def main(args):
    print(args)
    start = time.perf_counter()
    if args.query is not None:
        query = args.query
    else:
        if args.database == "p":
            query = queries_postgres[args.preset]

        else:
            query = queries_neo4j[args.preset]
    if args.args:
        for i, val in enumerate(args.args):
            query = query.replace(f"REPL{i}", val)

    spark = init_sparksession(query, db=args.database)
    spark.show(
    )
    if args.save:
        spark.write.csv(path=f"./results/{args.save}.csv")
    end = time.perf_counter()
    print(f"Took {end - start}")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q",   "--query",                              help="enter SQL/cypher query",           type=str)
    parser.add_argument("-p", "--preset",                           help="choose predefined query",   type=str)
    parser.add_argument("-s",   "--save",          help="Save to /results. Add a filename as an argument", type=str)
    parser.add_argument("-d",  "--database",                           help="p=postgres or n=neo4j")
    parser.add_argument("-a", "--args", nargs='+', help="custom arguments such as blocknumber")
    return parser.parse_args()





if __name__ == "__main__":
    arguments = argparser()

    if arguments.query is None and arguments.preset is None:
        raise UserWarning("A predefined or userdefined query via flags {-q, -pre} is required")
        exit()
    if arguments.query is not None and arguments.preset is not None:
        raise UserWarning("Cannot process predefined AND userdefined query. Select one and remove other")
        exit()

    main(arguments)







