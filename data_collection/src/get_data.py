import json
import ssl
from substrateinterface import SubstrateInterface
from collections import OrderedDict
import subprocess
from src.utils import progress_of_loop


class StakingSnapshot:
    """
    Creates Snapshot at block_number.
    If none exists it calls various storage queries to generate custom snapshot.
    """

    def __init__(self):
        self.substrate = None
        self.block_hash = None
        self.block_number = None
        self.snapshot = None
        self.era = None
        self.weird_accounts = [
            "1LMT8pQSCetUTQ6Pz1oMZrT1iyBBu6pYgo5C639NEF9utRu",
            "14iL6YGYwXsDNvmu78Uj8EJRuNiRzq4Uw6wraszet2qkeb8q",
            "16gKSjGn6UBgj4ZTm9GFKtvi1Zq7rhaB9XZZNsuAaEEvKwfQ",
            "15MV2Bj8DC5yCw1FJBoBXKzFLYLqfyvwjfwctGZqHUNsfBiN",
            "14JsCXA54ZzXFkLFUuXEKSjX6zD7f4pGZmCS9L5Mjcq2aWHY",
            "135yiqWJiaixaWPyEwyFZs4eaBDsmGqAunJpBot3mxQquYpG",
            "11uywEbA2VgRimPLqL8tTNWErDE7sZXe7aycnErJu28bBTx",
        ]
        self.current_nominator_max = 22500

    def create_substrate_connection(self, config_path):
        if config_path is None:
            raise UserWarning("Must provide valid config json")
        with open(config_path, "r") as f:
            node_config = json.loads(f.read())["node"]
        sslopt = {"sslopt": {"cert_reqs": ssl.CERT_NONE}}
        substrate = SubstrateInterface(
            url=node_config["url"],
            ss58_format=node_config["ss58_format"],
            type_registry_preset=node_config["type_registry_preset"],
            ws_options=sslopt,
        )
        self.substrate = substrate
        return substrate

    def set_block_number(self, block_number):
        self.block_number = block_number
        self.block_hash = self.get_blockhash_from_blocknumber(self.block_number)
        self.era = str(self.get_era()["index"])

    def set_era(self, era):
        self.era = str(era)

    def query(self, module, storage_function, parameters, block_hash):
        return self.substrate.query(
            module=module,
            storage_function=storage_function,
            params=parameters,
            block_hash=block_hash,
        ).value

    def query_map(self, module, storage_function, parameters, block_hash):
        return self.substrate.query_map(
            module=module,
            storage_function=storage_function,
            block_hash=block_hash,
            params=parameters
            # not sure what value makes sense, this worked so far.
        )

    def get_validator_exposure(self, era):
        result = self.query_map(
            module="Staking",
            storage_function="ErasStakers",
            parameters=[
                era,
            ],
            block_hash=self.block_hash,
        )
        exposure_dict = {}
        for row in result.records:
            exposure_dict[row[0].value] = row[1].value
        return exposure_dict

    def get_era(self, block_hash=None):
        if self.block_hash is None:
            block_hash = block_hash
        else:
            block_hash = self.block_hash
        return self.query(
            module="Staking",
            storage_function="ActiveEra",
            parameters=[],
            block_hash=block_hash,
        )

    def get_blockhash_from_blocknumber(self, block_number):
        return self.substrate.get_block_hash(block_number)

    def get_specific_nominator_vote(self, address):
        targets = self.query(
            module="Staking",
            storage_function="Nominators",
            parameters=[address],
            block_hash=self.block_hash,
        )
        if targets is None:
            return [address]
        else:
            return targets["targets"]

    def get_specific_nominator_exposure(self, address):
        locked_balance = self.query(
            module="Balances",
            storage_function="Locks",
            parameters=[address],
            block_hash=self.block_hash,
        )
        for balance in locked_balance:
            if (
                balance["id"] == "0x7374616b696e6720"
            ):  # "0x7374616b696e6720" = 'staking'
                return balance["amount"]

    def get_voterlist_bags(self):
        return self.query_map(
            module="VoterList",
            storage_function="ListBags",
            parameters=[],
            block_hash=self.block_hash,
        )

    def get_voterlist_neighbour(self, address):
        return self.query(
            module="VoterList",
            storage_function="ListNodes",
            parameters=[address],
            block_hash=self.block_hash,
        )

    def get_account_indices(self):
        return self.query_map(
            module="Indices",
            storage_function="Accounts",
            parameters=[],
            block_hash=self.block_hash,
        )

    def get_targets(self):
        target_dict = self.query_map(
            module="Staking",
            storage_function="ErasValidatorPrefs",
            parameters=[self.era],
            block_hash=self.block_hash,
        )
        targets = []
        for target in target_dict:
            targets.append(target[0].value)
        return targets

    def get_snapshot(self):
        # print(f'attempting snapshot query at {self.block_hash, self.era, self.block_number}')

        substrate_snapshot = self.query(
            module="ElectionProviderMultiPhase",
            storage_function="Snapshot",
            parameters=[],
            block_hash=self.block_hash,
        )
        if substrate_snapshot is not None:
            # print("Substrate Snapshot available :)")
            return substrate_snapshot
        else:
            self.get_historical_snapshot()

    def get_stored_solution(self):
        return self.query(
            module="ElectionProviderMultiPhase",
            storage_function="SignedSubmissionsMap",
            parameters=[0],
            block_hash=self.block_hash,
        )

    def calculate_optimal_solution(self, path_to_snapshot, iterations="10"):
        path_to_snapshot_file = path_to_snapshot + str(self.era) + "_snapshot.json"
        result = subprocess.run(
            [
                "../hackingtime/target/debug/sequential_phragmen_custom",
                path_to_snapshot_file,
                iterations,
                str(self.era),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        # Extract the output of the Rust script
        output = result.stdout

        # Split the output into two strings
        string_winners, string_assignments = output.strip().split("  ")

        # Parse the JSON strings into Python objects
        json_winners = json.loads(string_winners)
        json_assignments = json.loads(string_assignments)

        return json_winners, json_assignments

    def write_to_json(self, name, data_to_save, storage_path):
        file_path = storage_path + str(self.era) + name
        with open(file_path, "w", encoding="utf-8") as jsonfile:
            jsonfile.write(json.dumps(data_to_save, ensure_ascii=False, indent=4))
            jsonfile.close()

    def get_historical_snapshot(self):
        targets = self.get_targets()
        voter_pointers_dict = self.__transform_to_ordereddict(self.get_voterlist_bags())
        full_voterlist = []
        bagscounter = 0
        for bag in voter_pointers_dict:
            progress_of_loop(bagscounter, voter_pointers_dict, "Bags")
            if len(full_voterlist) == self.current_nominator_max:
                break
            head = voter_pointers_dict[bag]["head"]
            full_voterlist.append(head)
            tail = voter_pointers_dict[bag]["tail"]
            if head == tail:
                continue
            else:
                while head != tail:
                    if len(full_voterlist) == self.current_nominator_max:
                        break
                    head = self.get_voterlist_neighbour(head)["next"]
                    if head not in self.weird_accounts:
                        full_voterlist.append(head)
        voters = []
        counter = 0
        for voter in full_voterlist:
            progress_of_loop(counter, full_voterlist, "Nominators")
            nominator = []
            bond = self.get_specific_nominator_exposure(voter)
            specific_nominator_targets = self.get_specific_nominator_vote(voter)
            nominator.append(voter)
            nominator.append(bond)
            nominator.append(specific_nominator_targets)
            voters.append(nominator)
        return {"voters": voters, "targets": targets}

    @staticmethod
    def __transform_to_ordereddict(voterlist):
        voterdict = {}
        for row in voterlist:
            voterdict[row[0].value] = row[1].value
        sortedbags = sorted(voterdict.keys(), reverse=True)
        ordered_voterdict = OrderedDict()
        for bag in sortedbags:
            ordered_voterdict[bag] = voterdict[bag]
        return ordered_voterdict


if __name__ == "__main__":
    snapshot = StakingSnapshot()
    snapshot.create_substrate_connection(config_path="./config.json")
    snapshot.set_block_number(9771970)
    set1 = snapshot.query(
        module="Staking",
        storage_function="ErasRewardPoints",
        parameters=[968],
        block_hash=snapshot.block_hash,
    )

    set2 = snapshot.query(
        module="Staking",
        storage_function="ErasRewardPoints",
        parameters=[967],
        block_hash=snapshot.block_hash,
    )

    list1 = set()
    for row in set1["individual"]:
        list1.add(row[0])
    list2 = set()
    for row in set2["individual"]:
        list2.add(row[0])
    diff = len(list1.difference(list2))
    print()

    """   
    snapshot = StakingSnapshot()
    snapshot.create_substrate_connection(config_path="./config.json")
    era = str(snapshot.get_era()['index'])

    what = snapshot.get_validator_exposure(986)
    print()
    """

    # snapshot.calculate_optimal_solution()

    """
    snapshot_instance = StakingSnapshot(config_path='config.json')

    winners, assignments = snapshot_instance.calculate_optimal_solution("./snapshot_data/590_custom_snapshot.json",
                                                                        "10000")
    snapshot_instance.write_to_json(name="_winners.json",
                                    data_to_save=winners,
                                    storage_path="./calculated_solutions/")
    snapshot_instance.write_to_json(name="_assignments.json",
                                    data_to_save=assignments,
                                    storage_path="./calculated_solutions/")
    snapshot_utility = StakingSnapshotUtility(snapshot_instance)
    df = snapshot_utility.group_data("./block_numbers/signedphase_blocknumer.json",
                                     "./block_numbers/solutionstored_blocknumbers.json")
    snapshot_utility.save_file(df)
    snapshot_instance.set_block_number(13946907)
    indexes = snapshot_instance.get_account_indices()"""

    """
    from os import listdir
    from os.path import isfile, join
    import pandas as pd

    snapshot_instance = StakingSnapshot()
    snapshot_instance.create_substrate_connection("config.json")
    path = 'block_numbers/signedphase_blocknumbers/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    block_numbers = []
    for file in files:
        data = pd.read_csv(path + file)
        for value in data.values:
            if isinstance(value[1], int):
                block_numbers.append(value[1])

    with open("./block_numbers/signedphase_blocknumbers.json", 'w', encoding='utf-8') as jsonfile:
        jsonfile.write(json.dumps(block_numbers, ensure_ascii=False, indent=4))
        jsonfile.close()
"""
    """
    blockcounter = 0
    for number in sorted(block_numbers):
        print(number)
        blockcounter += 1
        blockprogress = (blockcounter / len(block_numbers)) * 100
        sys.stdout.write("Blocknumber Progress: %d%%   \r" % blockprogress)
        sys.stdout.flush()
        snapshot_instance = StakingSnapshot(config_path='../config.json', block_number=number)
        snapshot, index = snapshot_instance.get_stored_solution()
        if snapshot is not None:
            snapshot_instance.write_to_json('_' + str(index) + '_storedsolution_.json', snapshot)
    blockcounter
    """
    """
    #with open("signedphase_blocknumbers.json", "r") as jsonfile:
        blocknumbers = json.load(jsonfile)

    from os import listdir
    from os.path import isfile, join
    import re
    mypath = './data/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    covered_eras=[]
    for txt in onlyfiles:
        covered_eras.append(re.findall(r'\d+', txt))
    era_compare = []
    for era in covered_eras:
        era_compare.append(int(era[0]))
    blockcounter = 0
    for number in sorted(blocknumbers):
        print(number)
        blockcounter += 1
        blockprogress = (blockcounter/len(blocknumbers))*100
        sys.stdout.write("Blocknumber Progress: %d%%   \r" % blockprogress)
        sys.stdout.flush()
        snapshot_instance = StakingSnapshot(config_path='config.json', block_number=number)
        era = snapshot_instance.get_era()
        if era['index'] not in era_compare:
            snapshot = snapshot_instance.get_snapshot()
            snapshot_instance.write_to_json('_custom_snapshot.json', snapshot)
        else:
            print(f'already done : {era["index"]}')
    blockcounter
    """
