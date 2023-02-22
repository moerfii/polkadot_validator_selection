import os
import time

from get_data import StakingSnapshot
from preprocessor import Preprocessor

snapshot = StakingSnapshot()
substrate = snapshot.create_substrate_connection(config_path="../config.json")
snapshot_path = "../data/snapshot_data/"

def subscription_handler(account_info_obj, update_nr, subscription_id):

    if update_nr == 0:
        print('Initial account data:', account_info_obj.value)

    if update_nr > 0:
        snapshot.era = snapshot.get_era()['index']
        snapshot_data = account_info_obj.value
        nominator_mapping, validator_mapping = Preprocessor().return_mapping_from_address_to_index(snapshot_data)
        snapshot.write_to_json(name="_snapshot.json",
                                        data_to_save=snapshot_data,
                                        storage_path=snapshot_path)
        snapshot.write_to_json(name="_snapshot_nominator_mapping.json",
                                        data_to_save=nominator_mapping,
                                        storage_path=snapshot_path)
        snapshot.write_to_json(name="_snapshot_validator_mapping.json",
                                        data_to_save=validator_mapping,
                                        storage_path=snapshot_path)
        # Do something with the update
        print('Account data changed:', account_info_obj.value)

    # The execution will block until an arbitrary value is returned, which will be the result of the `query`
    if update_nr > 5:
        return account_info_obj


result = substrate.query("ElectionProviderMultiPhase", "Snapshot", [],
                         subscription_handler=subscription_handler)

"""             
prev_block = 0
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
"""
