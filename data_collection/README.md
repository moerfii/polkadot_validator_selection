Please supply a config.json file with following structure:

{
  "node": {
    "url": "",                          # provide the address to your node
    "ss58_format": 0,                   
    "type_registry_preset": "polkadot"
  }
}

Requires a running node with RPC enabled. Public nodes are not sufficient due to limiting response sizes.
Create new environment with python3 -m venv env
Activate environment with source env/bin/activate
Install requirements with make install
make sure to set up rust and cargo, build the application in the hackingtime folder with cargo build
then simply run the python script with python3 main.py -p <path to config.json> -m <mode>

