use std::fs;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::Write;
use sp_arithmetic::PerThing;
use sp_arithmetic::Perbill;
use itertools::Itertools;


#[derive(Serialize, Deserialize, Debug)]
struct Voter {
    name: String,
    bond: u64,
    targets: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Snapshot {
    voters: Vec<(String, u64, Vec<String>)>,
    targets: Vec<String>,
}


#[derive(Serialize, Deserialize, Debug)]
struct VoterDistribution<AccountId, P: PerThing> {
    voters: Vec<AccountId>,
    distributions: Vec<(AccountId, P)>,
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "std", derive(PartialEq, Eq, Encode, Decode))]
pub struct Assignment<AccountId, P: PerThing + Serialize> {
    /// Voter's identifier.
    pub who: AccountId,
    /// The distribution of the voter's stake.
    pub distribution: Vec<(AccountId, P)>,
}

fn main() {

    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        println!("Usage: {} <snapshot_filepath> <iterations> <era>", args[0]);
        return;
    }
    
    let snapshot_filepath = &args[1];
    let iterations = match args[2].parse::<usize>() {
        Ok(num) => num,
        Err(_) => {
            println!("Error: Second argument must be an integer.");
            return;
        }
    };
    let era = &args[3];

    let data = fs::read_to_string(&snapshot_filepath)
    .expect("Unable to read file");

    let json: Snapshot =  serde_json::from_str(&data)
    .expect("JSON does not have correct format.");


    let candidates = json.targets;
    let voters = json.voters;
    let config = sp_npos_elections::BalancingConfig{iterations:iterations,tolerance:10000};
	let sp_npos_elections::ElectionResult::<_, sp_runtime::Perbill> { winners, assignments } = sp_npos_elections::phragmms(
		297,
		candidates,
		voters,
		Some(config),
	)
	.unwrap();

    let grouped = assignments
        .into_iter()
        .group_by(|assignment| assignment.who.clone())
        .into_iter()
        .map(|(who, group)| {
            (
                who,
                group.flat_map(|assignment| assignment.distribution.into_iter()).collect(),
            )
        })
        .collect::<Vec<(String, Vec<(String, Perbill)>)>>();


    let json_winners = serde_json::to_string(&winners).unwrap();
    //let filename_winners = era.to_owned() + "_winners.json";
    //let mut file_winners = File::create(filename_winners).unwrap();
    //file_winners.write_all(json_winners.as_bytes()).unwrap();

    let json_voters = serde_json::to_string(&grouped).unwrap();
    //let filename_assignments = era.to_owned() + "_assignments.json";
    //let mut file_assignments = File::create(filename_assignments).unwrap();
    //file_assignments.write_all(json_voters.as_bytes()).unwrap();

    println!("{}  {}",json_winners, json_voters);
}

