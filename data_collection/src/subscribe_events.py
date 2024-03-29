from get_data import StakingSnapshot
from preprocessor import Preprocessor
from utils import save_json


def subscription_handler(account_info_obj, update_nr, subscription_id):
    if update_nr == 0:
        print("Initial account data:", account_info_obj.value)

    if update_nr > 0:
        era = str(snapshot.get_era()["index"])
        snapshot_data = account_info_obj.value

        snapshot_path = "../data/snapshot_data/"
        snapshot_file_path = snapshot_path + era + "_snapshot.json"
        snapshot_nominator_mapping_file_path = (
            snapshot_path + era + "_snapshot_nominator_mapping.json"
        )
        snapshot_validator_mapping_file_path = (
            snapshot_path + era + "_snapshot_validator_mapping.json"
        )
        (
            nominator_mapping,
            validator_mapping,
        ) = Preprocessor().return_mapping_from_address_to_index(snapshot_data)
        save_json(snapshot_file_path, snapshot_data)
        save_json(snapshot_nominator_mapping_file_path, nominator_mapping)
        save_json(snapshot_validator_mapping_file_path, validator_mapping)
        # Do something with the update
        print("Account data changed:", account_info_obj.value)

    # The execution will block until an arbitrary value is returned, which will be the result of the `query`
    if update_nr > 5:
        return account_info_obj


if __name__ == "__main__":
    snapshot = StakingSnapshot()
    substrate = snapshot.create_substrate_connection(config_path="../config.json")
    result = substrate.query(
        "ElectionProviderMultiPhase",
        "Snapshot",
        [],
        subscription_handler=subscription_handler,
    )

