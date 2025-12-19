# Updated simulator with XGBoost

from main import Card
from main import Player
from main import TerraformingMarsSoloGame
from collections import deque
import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle


def progress_score(state):
    T = state["temperature"]
    O = state["oxygen"]
    W = state["oceans"]

    progress = (T + 30)/38 + O/14 + W/9
    return progress/3.000     # normalize 0-1

weights = {
    "megacredits_production": 0.07,   # production
    "steel_production": 0.05,
    "titanium_production": 0.05,
    "plants_production": 0.07,
    "energy_production": 0.07,
    "heat_production": 0.08,
    "gain_megacredits": 0.07,   # gain resources
    "gain_steel": 0.05,
    "gain_titanium": 0.05,
    "gain_plants": 0.07,
    "gain_energy": 0.07,
    "gain_heat": 0.08,
    "increase_TR": 0.07,
    "tag_building": 0.02,       # tags
    "tag_space": 0.02,          
    "tag_science": 0.03,        
    "tag_power": 0.02,          
    "tag_earth": 0.01,          
    "tag_jovian": 0.01,         
    "tag_plant": 0.02,          
    "tag_microbe": 0.01,        
    "tag_city": 0.01,           
    "tag_event": 0.01,
    "tag_standard": 0.01  
}

weight_progress = 1.1
weight_tags = 0.01
weight_syn = 0.05
# value of tags
def valuable_tags(card):
    return (
        1 * card["tag_science"] +
        1 * card["tag_plant"] +
        1 * card["tag_power"] + 
        1 * card["tag_space"]
    )

# value of syngergies and states
def synergy(card, state):
    return card["tag_science"] + state["tag_science"]

def compute_ELTC(card, state=None):
    sum = 0
    # for production and tags
    for key, value in weights.items():
        sum += card[key] * value
    # for tags and synergy
    sum += weight_tags * valuable_tags(card)
    # sum += weight_syn * synergy(card, state)

    return sum

def encode_state(global_parameters, generation, terraform_rating, production, resources, resource_value, played_tags):
    return {
        "temperature": global_parameters.get("temperature"),
        "oxygen": global_parameters.get("oxygen"),
        "oceans": global_parameters.get("oceans"),
        "generation": generation,
        "terraform_rating": terraform_rating,

        "megacredits_production": production.get("megacredits"),
        "steel_production": production.get("steel"),
        "titanium_production": production.get("titanium"),
        "plants_production": production.get("plants"), 
        "energy_production": production.get("energy"),
        "heat_production": production.get("heat"),

        "megacredits": resources.get("megacredits"),
        "steel": resources.get("steel"),
        "titanium": resources.get("titanium"),
        "plants": resources.get("plants"),
        "energy": resources.get("energy"),
        "heat": resources.get("heat"),

        "steel_value": resource_value.get("steel"),
        "titanium_value": resource_value.get("titanium"),

        "tag_building": played_tags.get("building"),
        "tag_space": played_tags.get("space"),
        "tag_science": played_tags.get("science"),
        "tag_power": played_tags.get("power"),
        "tag_earth": played_tags.get("earth"),
        "tag_jovian": played_tags.get("jovian"),
        "tag_plant": played_tags.get("plant"),
        "tag_microbe": played_tags.get("microbe"),
        "tag_city": played_tags.get("city"),
        "tag_event": played_tags.get("event"),
        "tag_standard": played_tags.get("Standard")

    }

def encode_card(name, cost, effects, tags, conditions):
    return {
        "name": name,
        
        "cost": cost,

        "megacredits_production": effects.get("megacredits_production", 0),
        "steel_production": effects.get("steel_production", 0),
        "titanium_production": effects.get("titanium_production", 0),
        "plants_production": effects.get("plants_production", 0),
        "energy_production": effects.get("energy_production", 0),
        "heat_production": effects.get("heat_production", 0),
        "gain_megacredits": effects.get("gain_megacredits", 0),
        "gain_steel": effects.get("gain_steel", 0),
        "gain_titanium": effects.get("gain_titanium", 0),
        "gain_plants": effects.get("gain_plants", 0),
        "gain_energy": effects.get("gain_energy", 0),
        "gain_heat": effects.get("gain_heat", 0),
        "steel_value": effects.get("resource_value_steel", 0),
        "titanium_value": effects.get("resource_value_titanium", 0),
        "increase_TR": effects.get("increase_TR", 0),

        "tag_building": tags.count("building"),
        "tag_space": tags.count("space"),
        "tag_science": tags.count("science"),
        "tag_power": tags.count("power"),
        "tag_earth": tags.count("earth"),
        "tag_jovian": tags.count("jovian"),
        "tag_plant": tags.count("plant"),
        "tag_microbe": tags.count("microbe"),
        "tag_city": tags.count("city"),
        "tag_event": tags.count("event"),
        "tag_standard": tags.count("Standard"),

        "increase_temperature": effects.get("increase_temperature", 0),
        "increase_oxygen": effects.get("increase_oxygen", 0),
        "place_ocean": effects.get("place_ocean", 0),

        "requires_min_temperature": conditions.get("global_params_min", {}).get("temperature", -30),
        "requires_min_oxygen": conditions.get("global_parmas_min", {}).get("oxygen", 0),
        "requires_min_oceans": conditions.get("global_parmas_min", {}).get("oceans", 0),
        "requires_max_temperature": conditions.get("global_params_max", {}).get("temperature", 8),
        "requires_max_oxygen": conditions.get("global_params_max", {}).get("oxygen", 14),
        "requires_max_oceans": conditions.get("global_params_max", {}).get("oceans", 9),
        "requires_tag_jovian": conditions.get("tags_played", {}).get("jovian", 0),
        "requires_tag_power": conditions.get("tags_played", {}).get("power", 0),
        "requires_tag_science": conditions.get("tags_played", {}).get("science", 0),
        "requires_steel_production": conditions.get("production", {}).get("steel", 0),
        "requires_titanium_production": conditions.get("production", {}).get("titanium", 0),
        "requires_plants_production": conditions.get("production", {}).get("plants", 0),
        "requires_energy_production": conditions.get("production", {}).get("energy", 0),
        "requires_heat_production": conditions.get("production", {}).get("heat", 0),
        "requires_plants": conditions.get("resources", {}).get("plants", 0),
        "requires_energy": conditions.get("resources", {}).get("energy", 0),

    }


def play_card(card, old_state):

    new_state = old_state

    for key, value in card.items():
        # if there's a requirement to play the card, check if it can be played first
        if key.startswith("requires_"):
            condition = key.replace("requires_", "")

            # if it's related to global requirements
            if condition.startswith("min_"):
                global_parameter = condition.replace("min_", "")
                if (new_state.get(global_parameter) < value):
                    return new_state

            elif condition.startswith("max_"):
                global_parameter = condition.replace("max_", "")
                if (new_state.get(global_parameter) > value):
                    return new_state
                
            # if it's related to tags played
            elif condition.startswith("tag_"):
                if (new_state.get(condition) < value):
                    return new_state
            
            # if it's related to production of resources
            elif condition.endswith("_production"):
                if (new_state.get(condition) < value):
                    return new_state
                
            # otherwise related to resources needed
            else:
                if (new_state.get(condition) < value):
                    return new_state
    
    net_cost = card["cost"]
    
    steel_used = 0
    titanium_used = 0
    
    # determine if card can be paid with megacredits and/or steel/titanium depending on tags
    if card["tag_building"] or card["tag_space"]:
        
        if card["tag_space"]:
            # resource needed is titanium
            titanium_value = new_state["titanium_value"]
            titanium_max = new_state["titanium"]

            titanium_needed = (net_cost - titanium_value - 1) // titanium_value
            titanium_used = min(titanium_max, titanium_needed)
            titanium_discount = titanium_used * titanium_value
            net_cost = max(0, titanium_discount)
        
        if card["tag_building"]:
            # resource needed is steel
            steel_value = new_state["steel_value"]
            steel_max = new_state["steel"]

            steel_needed = (net_cost - steel_value - 1) // steel_value
            steel_used = min(steel_max, steel_needed)
            steel_discount = steel_used * steel_value
            net_cost = max(0, steel_discount)

    # check if can afford to pay for the card
    if new_state["megacredits"] < net_cost:
        return new_state

    # pay for card
    new_state["megacredits"] -= net_cost
    new_state["steel"] -= steel_used
    new_state["titanium"] -= titanium_used

    # only apply effects where the values are non-zero, is not name or cost, and doesn't have a requirement
    filtered = {
        key: value
        for key, value in card.items()
        if value != 0
        and key not in {"name", "cost"}
        and not key.startswith("requires_")
    }

    effects_deque = deque(filtered.items())

    # modify effects one at a time
    while effects_deque:
        effect = effects_deque.popleft()
        effect_name = effect[0]
        effect_value = effect[1]

        if effect_name.endswith("_production"):
            new_state[effect_name] += effect_value

        elif effect_name.endswith("_value"):
            new_state[effect_name] += effect_value

        elif effect_name.startswith("gain_"):
            resource = effect_name.replace("gain_", "")
            new_state[resource] += effect_value
        
        elif effect_name.startswith("tag_"):
            new_state[effect_name] += effect_value
        
        elif effect_name == "increase_TR":
            new_state["terraform_rating"] += effect_value

        # global terraform rating change
        else:
            # raise oxygen effect
            if effect_name == "raise_oxygen":
                old_oxygen = new_state["oxygen"]
                new_state["oxygen"] = min(14, new_state["oxygen"] + effect_value)
                net_increase_oxy = new_state["oxygen"] - old_oxygen
                new_state["terraform_rating"] += net_increase_oxy

                # if oxygen level hits 8, increase temperature
                if old_oxygen < 8 <= new_state["oxygen"]:
                    effects_deque.appendleft(("increase_temperature", 1))

            # raise temperature effect
            if effect_name == "increase_temperature":
                old_temperature = new_state["temperature"]
                new_state["temperature"] = min(8, new_state["temperature"] + effect_value * 2)
                net_increase_step = (new_state["temperature"] - old_temperature) // 2
                new_state["terraform_rating"] += net_increase_step

                # if temperature hits -24 or -20, increase heat production
                if old_temperature < -24 <= new_state["temperature"]:
                    new_state["heat_production"] += 1
                if old_temperature < -20 <= new_state["temperature"]:
                    new_state["heat_production"] += 1
                
                # place ocean at 0
                if old_temperature < 0 <= new_state["temperature"]:
                    effects_deque.appendleft(("place_ocean", 1))

            # place ocean effect
            if effect_name == "place_ocean":
                old_oceans = new_state["oceans"]
                new_state["oceans"] = min(9, new_state["oceans"] + effect_value)
                net_increase_oceans = new_state["oceans"] - old_oceans
                new_state["terraform_rating"] += net_increase_oceans
        
    return new_state

def build_training_row(old_state: dict, card: dict):
    # determine new state of game after card is played
    new_state = play_card(card, old_state)

    # combine dictionaries into a single feature row
    x_row = {
        **{f"old_{k}": v for k, v in old_state.items()},
        **{f"card_{k}": v for k, v in card.items()},
        **{f"new_{k}": v for k, v in new_state.items()},
    }

    # compute target
    y_value = (
        weight_progress
        * (progress_score(new_state) - progress_score(old_state))
        + compute_ELTC(card)
    )

    return x_row, y_value

# define generation progress 
def generation_from_progress(progress):
    if progress < 0.10:
        lo, hi = 1, 2
    elif progress < 0.25:
        lo, hi = 2, 4
    elif progress < 0.45:
        lo, hi = 4, 6
    elif progress < 0.65:
        lo, hi = 6, 8
    elif progress < 0.85:
        lo, hi = 8, 10
    else:
        lo, hi = 10, 12

    return np.random.randint(lo, hi + 1)


# generate state from random numbers
def generate_random_state():

        gp_rand = {
            "oxygen": np.random.randint(0, 14),
            "temperature": np.random.randint(-15, 4)*2,
            "oceans": np.random.randint(0, 9)
        }
        tr_rand = gp_rand["oxygen"] + gp_rand["oceans"] + int((30 + gp_rand["temperature"])/2) + 14 + np.random.randint(0, 5)
        gen_rand = generation_from_progress(progress_score(gp_rand))

        prod_rand = {
            "megacredits": np.random.randint(-2, 10),
            "steel": np.random.randint(0, 5),
            "titanium": np.random.randint(0, 4),
            "plants": np.random.randint(0, 5),
            "energy": np.random.randint(0, 6),
            "heat": np.random.randint(0, 10)
        }

        res_rand = {
            "megacredits": max(0, np.random.randint(0, tr_rand) + np.random.randint(-10, 10)),
            "steel": np.random.randint(0, 4),
            "titanium": np.random.randint(0, 4),
            "plants": np.random.randint(0, 8),
            "energy": np.random.randint(0, prod_rand["energy"]+1),
            "heat": np.random.randint(0, 8)
        }

        res_value_rand = {
            "steel": 2,
            "titanium": 3
        }

        tags_rand = {
            "building": np.random.randint(0, 3),
            "space": np.random.randint(0, 3),
            "science": np.random.randint(0, 8),
            "power": np.random.randint(0, 3),
            "earth": np.random.randint(0, 2),
            "jovian": np.random.randint(0, 2),
            "plant": np.random.randint(0, 4),
            "microbe": np.random.randint(0, 2),
            "city": np.random.randint(2, 5),
            "event": np.random.randint(0, 4),
            "Standard": np.random.randint(0, 9)
        }

        # returns a dataframe of the entire state
        return encode_state(gp_rand, gen_rand, tr_rand, prod_rand, res_rand, res_value_rand, tags_rand)

# build dataset from random states
def build_dataset(deck, n_samples):
    X_rows = []
    Y_rows = []

    for _ in range(n_samples):
        old_state_vec = generate_random_state()
        card = random.choice(deck)
        card_vec = encode_card(
            card.name, card.cost, card.effects, card.tags, card.conditions
        )

        x_row, y_value = build_training_row(old_state_vec, card_vec)

        X_rows.append(x_row)
        Y_rows.append({"target": y_value})

    X = pd.DataFrame(X_rows)
    Y = pd.DataFrame(Y_rows)

    return X, Y

if __name__ == "__main__":
    # game instance for training
    trainergame = TerraformingMarsSoloGame()

    # parameters needed for the training states
    global_parameters = trainergame.global_parameters
    generation = trainergame.generation
    resources = trainergame.player.resources
    production = trainergame.player.production
    resource_value = trainergame.player.resource_value
    played_tags = trainergame.player.played_tags
    terraform_rating = trainergame.player.terraform_rating

    deck = trainergame.generate_deck(played_tags)

    X, Y = build_dataset(deck, 3)



"""
def generate_simplified_game_state(n_samples=10000):
    return pd.DataFrame({
        'generation': np.random.randint(1, 15, n_samples),
        'megacredits': np.random.randint(0, 100, n_samples),
        'steel_production': np.random.randint(0, 6, n_samples),
        'energy_production': np.random.randint(0, 5, n_samples),
        'played_science_tags': np.random.randint(0, 10, n_samples),
        'terraform_rating': np.random.randint(20, 50, n_samples),
        'temperature': np.random.randint(-15, 4, n_samples)*2,
        'oxygen': np.random.randint(0, 14, n_samples),
        'oceans': np.random.randint(0, 9, n_samples)
    })

def generate_random_card(n_samples=10000):
    return pd.DataFrame({
        'card_cost': np.random.randint(5, 30, n_samples),
        'card_vp': np.random.randint(0, 3, n_samples),
        'effect_type': np.random.randint(0, 3, n_samples),
        'requires_science_tag': np.random.choice([0, 1], n_samples)
    })


def compute_synthetic_score(state_df, card_df):
    base_score = (
        state_df['terraform_rating'] +
        card_df['card_vp'] * 3 +
        state_df['steel_production'] +
        state_df['energy_production']
    )

    board_bonus = np.where(
        (card_df['effect_type'] == 1) & (state_df['temperature'] < 0), 5, 0
    )

    penalty = np.where(
        (card_df['requires_science_tag'] == 1) & (state_df['played_science_tags'] < 2), -5, 0
    )

    noise = np.random.normal(0, 2, len(base_score))

    return base_score + board_bonus + penalty + noise


def train_model():
    print("Generating training data...")
    n_samples = 10000
    state_df = generate_simplified_game_state(n_samples)
    card_df = generate_random_card(n_samples)
    X = pd.concat([state_df, card_df], axis=1)
    y = compute_synthetic_score(state_df, card_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Training XGBoost model...")
    model = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Model trained. Test RMSE: {rmse:.2f}")
    return model


def evaluate_cards(model, num_cards=3):
    print("\nEvaluating card values in a single game state...")
    state = generate_simplified_game_state(1).iloc[0:1].reset_index(drop=True)
    cards = generate_random_card(num_cards).reset_index(drop=True)

    for i in range(num_cards):
        input_df = pd.concat([state, cards.iloc[i:i+1]], axis=1)
        predicted_score = model.predict(input_df)[0]
        print(f"Card {i+1}:")
        print(cards.iloc[i])
        print(f"Predicted Final Score if played: {predicted_score:.2f}\n")


if __name__ == "__main__":
    model = train_model()
    evaluate_cards(model)
"""

