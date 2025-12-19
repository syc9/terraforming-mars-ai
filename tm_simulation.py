import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate Simplified Game State
def generate_simplified_game_state(n_samples=10000):
    return pd.DataFrame({
        'generation': np.random.randint(1, 15, n_samples),
        'mega_credits': np.random.randint(0, 100, n_samples),
        'steel_production': np.random.randint(0, 6, n_samples),
        'energy_production': np.random.randint(0, 5, n_samples),
        'played_science_tags': np.random.randint(0, 10, n_samples),
        'terraform_rating': np.random.randint(20, 50, n_samples),
        'temperature': np.random.randint(-30, 8, n_samples),
        'oxygen': np.random.randint(0, 14, n_samples),
        'oceans': np.random.randint(0, 9, n_samples)
    })

# 2. Generate Simplified Card Representation
def generate_random_card(n_samples=10000):
    return pd.DataFrame({
        'card_cost': np.random.randint(5, 30, n_samples),
        'card_vp': np.random.randint(0, 3, n_samples),
        'effect_type': np.random.randint(0, 3, n_samples),
        'requires_science_tag': np.random.choice([0, 1], n_samples)
    })

# 3. Synthetic Scoring Function (for model training)
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

# 4. Training the Model
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

# 5. Evaluate Cards in a Given Game State
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

# Entry point
if __name__ == "__main__":
    model = train_model()
    evaluate_cards(model)
