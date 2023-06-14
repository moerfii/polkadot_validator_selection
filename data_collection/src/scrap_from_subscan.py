import argparse

import requests
import random
import time
import json

import substrateinterface

def get_events(args):
    url = "https://polkadot.api.subscan.io/api/scan/events"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0"
    }
    body = {
        "row": 100,
        "page": 1,
        "module": "electionprovidermultiphase",
        "event": "SignedPhaseStarted",
        #"address": "0x62Ec573d52ECc9cDD6ed93C00a2F85Bc657878B3",
        "block_range": f"{0}-{15000000}"
    }

    response = requests.post(url, json=body, headers=headers)

    le_json = []
    for i in range(0, 999):
        body["page"] = i
        print(f"Getting page {i}")
        response = requests.post(url, json=body, headers=headers)
        if len(response.json()["data"]) == 0:
            break
        if response.status_code == 200 and response.json()["data"]["events"] is not None:
            le_json = le_json + response.json()["data"]["events"]
            #random float between 0 and 2
            time.sleep(random.random()*2)
        else:
            print(f"Stopping due to {response.status_code} in page {i}")
            print(f"Got {len(le_json)} events")
            with open("../block_numbers/events.json", "w+") as f:
                f.write(json.dumps(le_json, indent=4))
            break


def get_block_numbers(args, snapshot_instance):
    snapshot_instance



    events = get_events(args)
    block_numbers = []
    for event in events:
        block_numbers.append(event["attributes"]["block_num"])


def setup():
    config = "config.json"
    with open(config, "r") as jsonfile:
        config = json.load(jsonfile)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    parser.add_argument("--start_block", type=int, required=True)
    parser.add_argument("--end_block", type=int, required=True)
    parser.add_argument("--block_numbers_path", type=str, required=True)
    return parser


if __name__ == "__main__":
    #parser = setup()
    #args = parser.parse_args()
    with open("../block_numbers/events.json", "r") as f:
        events = json.load(f)

    block_numbers = []
    for event in events:
        if event["event_id"] == "SignedPhaseStarted":
            block_numbers.append(event["block_num"])
    # write block numbers to file
    with open("../block_numbers/block_numbers.json", "w+") as f:
        f.write(json.dumps(block_numbers, indent=4))


