# Updated simulator with XGBoost

from main import Card
from main import Player
from main import TerraformingMarsSoloGame
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
    return progress/3.0     # normalize 0-1

weights = {
    "megacredits": 0.07,   # production
    "steel": .05,
    "titanium": 0.05,
    "plants": 0.07,
    "energy": 0.07,
    "heat": 0.08,
    "tag_building": 0.02,       # tags
    "tag_space": 0.02,          
    "tag_science": 0.03,        
    "tag_power": 0.02,          
    "tag_earth": 0.01,          
    "tag_jovian": 0.01,         
    "tag_plant": 0.02,          
    "tag_microbe": 0.01,        
    "tag_city": 0.01,           
    "tag_event": 0.01  
}

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

# value of syngergys and states
def synergy(card, state):
    return card["tag_science"] + state["tag_science"]

def compute_ELTC(card):
    sum = 0
    # for production and tags
    for key, value in weights:
        sum += card[key] * value
    # for tags and synergy
    sum += weight_tags * valuable_tags(card)
    sum += weight_syn * synergy(card, state=None)

    return sum

def encode_state(global_parameters, generation, production, resources, resource_value, played_tags):
    return pd.DataFrame({
        "temperature": global_parameters.get("temperature"),
        "oxygen": global_parameters.get("oxygen"),
        "oceans": global_parameters.get("oceans"),
        "generation": generation,

        "megacredits_prod": production.get("megacredits"),
        "steel_prod": production.get("steel"),
        "titanium_prod": production.get("titanium"),
        "plants_prod": production.get("plants"), 
        "energy_prod": production.get("energy"),
        "heat_prod": production.get("heat"),

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
        "tag_event": played_tags.get("event")

    })

def encode_card(name, cost, effects, tags, conditions):
    return pd.DataFrame({
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

        "tag_building": tags.get("building", 0),
        "tag_space": tags.get("space", 0),
        "tag_science": tags.get("science", 0),
        "tag_earth": tags.get("earth", 0),
        "tag_jovian": tags.get("jovian", 0),
        "tag_plant": tags.get("plant", 0),
        "tag_microbe": tags.get("microbe", 0),
        "tag_city": tags.get("city", 0),
        "tag_event": tags.get("event", 0),

        "increase_temperature": effects.get("increase_temperature", 0),
        "increase_oxygen": effects.get("increase_oxygen", 0),
        "place_ocean": effects.get("place_ocean", 0),

        "requires_min_temp": conditions.get("global_params_min", {}).get("temperature", -30),
        "requires_min_oxygen": conditions.get("global_parmas_min", {}).get("oxygen", 0),
        "requires_min_oceans": conditions.get("global_parmas_min", {}).get("oceans", 0),
        "requires_max_temp": conditions.get("global_params_max", {}).get("temperature", 8),
        "requires_max_oxygen": conditions.get("global_params_max", {}).get("oxygen", 14),
        "requires_max_oceans": conditions.get("global_params_max", {}).get("oceans", 9),
        "requires_tag_jovian": conditions.get("tags_played", {}).get("jovian", 0),
        "requires_tag_power": conditions.get("tags_played", {}).get("power", 0),
        "requires_tag_science": conditions.get("tags_played", {}).get("science", 0),
        "requires_steel_prod": conditions.get("production", {}).get("steel", 0),
        "requires_titanium_prod": conditions.get("production", {}).get("titanium", 0),
        "requires_plants_prod": conditions.get("production", {}).get("plants", 0),
        "requires_energy_prod": conditions.get("production", {}).get("energy", 0),
        "requires_heat_prod": conditions.get("production", {}).get("heat", 0),
        "requires_plants": conditions.get("resources", {}).get("plants", 0),
        "requires_energy": conditions.get("resources", {}).ge("energy", 0),

    })

# ------------------------
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
