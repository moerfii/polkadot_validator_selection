import time

from get_data import StakingSnapshot


def subscription_handler(obj, update_nr, subscription_id):
    if update_nr > 10:
        return {'message': 'Subscription will cancel when a value is returned', 'updates_processed': update_nr}
    return obj



snapshot = StakingSnapshot()
snapshot.create_substrate_connection(config_path="../config.json")

prev_block = 0
while True:
    result = snapshot.substrate.subscribe_block_headers(subscription_handler,
                                                        include_author=True,
                                                        finalized_only=False)
    if result['header']['number'] == prev_block:
        print("sleep")
        time.sleep(1)
    prev_block = result['header']['number']
    snapshot.set_block_number(result['header']['number'])
    hash = snapshot.get_blockhash_from_blocknumber(result['header']['number'])
    events = snapshot.substrate.get_events(block_hash=hash)
    for event in events:
        if event.value_serialized['module_id'] == "Electionprovidermultiphase" \
                and events.value_serialized['event_id'] == "SignedPhaseStarted":
            print("do something")

