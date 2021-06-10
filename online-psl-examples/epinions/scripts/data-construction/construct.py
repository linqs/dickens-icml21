import random
import os
import shutil

"""
Construct Online PSL commands for trust-prediction experiments.
These commands are all template modifications.
"""

# Hyperparameters for random model experiments
NUM_RANDOM_MODELS = 100
RANDOM_SEED = 2345

# Command constants.
WRITE_INFERRED_COMMAND = "WRITEINFERREDPREDICATES"
SYNC_COMMAND = "SYNC"
STOP_COMMAND = "STOP"
ADD_RULE_COMMAND = "ADDRULE"
DELETE_RULE_COMMAND = "DELETERULE"
ACTIVATE_RULE_COMMAND = "ACTIVATERULE"
DEACTIVATE_RULE_COMMAND = "DEACTIVATERULE"

# input/output paths
DIRNAME = os.path.dirname(__file__)
MODELS_PATH = os.path.join(DIRNAME, "../../cli/selected_models/")
RANDOM_MODELS_PATH = os.path.join(DIRNAME, "../../cli/random_models/")
COMMANDS_OUTPUT_PATH = os.path.join(DIRNAME, "../../data/epinions/commands/")

# set the seed globally
random.seed(RANDOM_SEED)

RULES = {
    # FF
    1: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    2: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    3: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    4: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    5: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    6: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    7: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    8: "1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    # FB
    9: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    10: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    11: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    12: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    13: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    14: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    15: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    16: "1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    # BF
    17: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    18: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    19: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    20: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    21: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    22: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    23: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    24: "1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    # BB
    25: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    26: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    27: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    28: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    29: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    30: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    31: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2",
    32: "1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2",
    # Symmetry
    33: "1.0: Knows(A, B) & Knows(B, A) & Trusts(A, B) -> Trusts(B, A) ^2",
    34: "1.0: Knows(A, B) & Knows(B, A) & !Trusts(A, B) -> !Trusts(B, A) ^2",
    # Two sided Prior
    35: "1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2",
    36: "1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2",
}

NEGATIVE_PRIOR = "0.1: !Trusts(A, B) ^2"


def create_single_random_model(rule_count):
    rules = ""
    rule_list = random.sample(RULES.keys(), rule_count)
    for rule_idx in rule_list:
        rules += RULES[rule_idx] + "\n"
    rules += NEGATIVE_PRIOR
    return rules


def create_random_models():
    # create random model list
    random_models = []

    # generate number of rules in each model
    rule_counts = [random.randint(1, len(RULES.keys())) for model_idx in range(NUM_RANDOM_MODELS)]

    # generate the random models
    for model_idx in range(NUM_RANDOM_MODELS):
        random_models += [create_single_random_model(rule_counts[model_idx])]

    # write models to disk and add them to MODELS map
    for model_idx, model in enumerate(random_models):
        model_filename = "epinions-random-" + str(model_idx) + ".psl"
        handle = open(RANDOM_MODELS_PATH + model_filename, "w+")
        handle.write(model)
        handle.close()
        MODELS["epinions-random-"+str(model_idx).zfill(2)] = {'filename': 'random_models/' + model_filename}


def create_full_model():
    # generate the random models
    full_model = create_single_random_model(len(RULES.keys()))

    # write models to disk and add them to MODELS map
    model_filename = "epinions-full.psl"
    handle = open(MODELS_PATH + model_filename, "w+")
    handle.write(full_model)
    handle.close()

# Assume start with full initial model
MODELS = {"FULL_MODEL":
              {"filename": "epinions.psl"},
          "balance-5":
              {"filename": "epinions-balance-5.psl"},
          "balance-5-recip":
              {"filename": "epinions-balance-5-recip.psl"},
          "balance-extended":
              {"filename": "epinions-balance-extended.psl"},
          "balance-extended-recip":
              {"filename": "epinions-balance-extended-recip.psl"},
          "status":
              {"filename": "epinions-status.psl"},
          "status-inv":
              {"filename": "epinions-status-inv.psl"},
          "cyclic-balanced":
              {"filename": "epinions-cyclic-balanced.psl"},
          "cyclic-balanced-unbalanced":
              {"filename": "epinions-cyclic-balanced-unbalanced.psl"}}


def construct_commands(models_directory):
    commands = []
    for model_file in os.listdir(models_directory):
        handle = open(os.path.join(models_directory, model_file), "r")
        rule_list = [rule.strip("\n") for rule in handle.readlines()]

        for rule in RULES:
            if RULES[rule] not in rule_list:
                commands += [DEACTIVATE_RULE_COMMAND + "\t" + RULES[rule]]

        commands += [WRITE_INFERRED_COMMAND + "\t\'./inferred-predicates/{}\'".format(model_file.split(".")[0])]

        for rule in RULES:
            if RULES[rule] not in rule_list:
                commands += [ACTIVATE_RULE_COMMAND + "\t" + RULES[rule]]

    commands += [STOP_COMMAND]

    return commands


def create_commands():
    # generate commands for random models
    commands = "\n".join(construct_commands(RANDOM_MODELS_PATH))
    command_file_handle = open(COMMANDS_OUTPUT_PATH + "random-commands.txt", "w+")
    command_file_handle.write(commands)
    command_file_handle.close()

    # generate commands for random models
    commands = "\n".join(construct_commands(MODELS_PATH))
    command_file_handle = open(COMMANDS_OUTPUT_PATH + "selected-commands.txt", "w+")
    command_file_handle.write(commands)
    command_file_handle.close()


def main():
    # delete existing random models if they exist
    if os.path.isdir(RANDOM_MODELS_PATH):
        shutil.rmtree(RANDOM_MODELS_PATH)

    # create fresh directory for random models
    os.mkdir(RANDOM_MODELS_PATH)

    # create the full model
    create_full_model()

    # create the random models
    create_random_models()

    # create command output dir if necessary
    if not os.path.isdir(COMMANDS_OUTPUT_PATH):
        os.mkdir(COMMANDS_OUTPUT_PATH)

    # create and write commands
    create_commands()


if __name__ == '__main__':
    main()