import json
import ssl
import sys
from substrateinterface import SubstrateInterface
from collections import OrderedDict


class StakingSnapshot:
    """
    Creates Snapshot at block_hash
    """

    def __init__(self, config_path, block_number):
        self.config_path = config_path
        self.substrate = self.__create_substrate_connection(config_path)
        self.block_hash = self.get_blockhash_from_blocknumber(block_number)
        self.block_number = block_number
        self.snapshot = None
        self.era = str(self.get_era()['index'])
        self.weird_accounts = ['1LMT8pQSCetUTQ6Pz1oMZrT1iyBBu6pYgo5C639NEF9utRu',
                               '14iL6YGYwXsDNvmu78Uj8EJRuNiRzq4Uw6wraszet2qkeb8q',
                               '16gKSjGn6UBgj4ZTm9GFKtvi1Zq7rhaB9XZZNsuAaEEvKwfQ',
                               '15MV2Bj8DC5yCw1FJBoBXKzFLYLqfyvwjfwctGZqHUNsfBiN',
                               '14JsCXA54ZzXFkLFUuXEKSjX6zD7f4pGZmCS9L5Mjcq2aWHY',
                               '135yiqWJiaixaWPyEwyFZs4eaBDsmGqAunJpBot3mxQquYpG',
                               '11uywEbA2VgRimPLqL8tTNWErDE7sZXe7aycnErJu28bBTx']
        self.current_nominator_max = 22500

    def get_era(self):
        return self.query(module='Staking', storage_function='ActiveEra', parameters=[], block_hash=self.block_hash)

    def query(self, module, storage_function, parameters, block_hash):
        return self.substrate.query(
            module=module,
            storage_function=storage_function,
            params=parameters,
            block_hash=block_hash
        ).value

    def query_map(self, module, storage_function, parameters, block_hash):
        return self.substrate.query_map(
            module=module,
            storage_function=storage_function,
            params=parameters,
            block_hash=block_hash,
            page_size=1000  # not sure what value makes sense, this worked so far.
        ).records

    @staticmethod
    def __create_substrate_connection(config_path):
        with open(config_path, "r") as f:
            node_config = json.loads(f.read())["node"]
        sslopt = {
            "sslopt": {
                "cert_reqs": ssl.CERT_NONE
            }
        }
        substrate = SubstrateInterface(
            url=node_config["url1"],  # todo: change to uni node
            ss58_format=node_config["ss58_format"],
            type_registry_preset=node_config["type_registry_preset"],
            ws_options=sslopt
        )
        return substrate

    def get_blockhash_from_blocknumber(self, block_number):
        return self.substrate.get_block_hash(block_number)

    def get_specific_nominator_vote(self, address):
        targets = self.query(module='Staking', storage_function='Nominators', parameters=[address],
                             block_hash=self.block_hash)
        if targets is None:
            return [address]
        else:
            return targets['targets']

    def get_specific_nominator_exposure(self, address):
        locked_balance = self.query(module='Balances', storage_function='Locks', parameters=[address],
                                    block_hash=self.block_hash)
        for balance in locked_balance:
            if balance['id'] == "0x7374616b696e6720":  # "0x7374616b696e6720" = 'staking'
                return balance['amount']

    def get_voterlist_bags(self):
        return self.query_map(module='VoterList', storage_function='ListBags', parameters=[],
                              block_hash=self.block_hash)  # must use query_map for this to work for some reason

    def get_voterlist_neighbour(self, address):
        return self.query(module='VoterList', storage_function='ListNodes', parameters=[address],
                          block_hash=self.block_hash)

    def get_account_indices(self):
        return self.query_map(module='Indices', storage_function='Accounts', parameters=[], block_hash=self.block_hash)

    def get_targets(self):
        target_dict = self.query_map(module='Staking', storage_function='ErasValidatorPrefs', parameters=[self.era],
                                     block_hash=self.block_hash)
        targets = []
        for target in target_dict:
            targets.append(target[0].value)
        return targets

    def get_snapshot(self):
        print(f'attempting snapshot query at {self.block_hash, self.era}')

        substrate_snapshot = self.query(module='ElectionProviderMultiPhase', storage_function='Snapshot', parameters=[],
                                        block_hash=self.block_hash)
        if substrate_snapshot is not None:
            print("Substrate Snapshot available :)")
            return substrate_snapshot
        else:
            self.get_historical_snapshot()


    def get_stored_solution(self):
        stored_solution_indices = self.query(module="ElectionProviderMultiPhase",
                                             storage_function="SignedSubmissionIndices",
                                             parameters=[], block_hash=self.block_hash)
        if not len(stored_solution_indices):
            return None, None
        for solution in stored_solution_indices:
            latest_solution_index = solution[1]
            solution = self.query(module="ElectionProviderMultiPhase",
                       storage_function="SignedSubmissionsMap",
                       parameters=[latest_solution_index], block_hash=self.block_hash)
            return solution, latest_solution_index
    def write_to_json(self, name, data_to_save):
        with open("./storedsolutions/" + self.era + name, 'w', encoding='utf-8') as jsonfile:
            json.dump(data_to_save, jsonfile, ensure_ascii=False, indent=4)

    def get_historical_snapshot(self):
        targets = self.get_targets()

        voter_pointers_dict = self.__transform_to_ordereddict(self.get_voterlist_bags())
        full_voterlist = []
        bagscounter = 0
        for bag in voter_pointers_dict:
            bagscounter += 1
            bagsprogress = (bagscounter / len(voter_pointers_dict)) * 100
            sys.stdout.write("Getting Accounts from Bags Progress: %d%%   \r" % bagsprogress)
            sys.stdout.flush()
            if len(full_voterlist) == self.current_nominator_max:
                break
            head = voter_pointers_dict[bag]['head']
            full_voterlist.append(head)
            tail = voter_pointers_dict[bag]['tail']
            if head == tail:
                continue
            else:
                while head != tail:
                    if len(full_voterlist) == self.current_nominator_max:
                        break
                    head = self.get_voterlist_neighbour(head)['next']
                    if head not in self.weird_accounts:
                        full_voterlist.append(head)
        voters = []
        counter = 0
        for voter in full_voterlist:
            counter += 1
            progress = (counter / self.current_nominator_max) * 100
            sys.stdout.write("Getting Bond and Targets Progress: %d%%   \r" % progress)
            sys.stdout.flush()
            nominator = []
            bond = self.get_specific_nominator_exposure(voter)
            specific_nominator_targets = self.get_specific_nominator_vote(voter)
            nominator.append(voter)
            nominator.append(bond)
            nominator.append(specific_nominator_targets)
            voters.append(nominator)
        return {'voters': voters, 'targets': targets}

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
    snapshot_instance = StakingSnapshot(config_path='config.json', block_number=13925946)
    indexes = snapshot_instance.get_account_indices()

    account_registry = {}
    for index in indexes:
        account_registry[index[0].value] = index[1].value[0]




    from os import listdir
    from os.path import isfile, join
    import pandas as pd

    path = './solutionstored_blocknumbers/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    block_numbers = []
    for file in files:
        data = pd.read_csv(path + file)
        for value in data.values:
            if isinstance(value[1], int):
                block_numbers.append(value[1])
    blockcounter = 0
    for number in sorted(block_numbers):
        print(number)
        blockcounter += 1
        blockprogress = (blockcounter / len(block_numbers)) * 100
        sys.stdout.write("Blocknumber Progress: %d%%   \r" % blockprogress)
        sys.stdout.flush()
        snapshot_instance = StakingSnapshot(config_path='config.json', block_number=number)
        snapshot, index = snapshot_instance.get_stored_solution()
        if snapshot is not None:
            snapshot_instance.write_to_json('_' + str(index) + '_storedsolution_.json', snapshot)
    blockcounter


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

