import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load and prepare data
# -----------------------------

data = pd.read_csv("data/tennis.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# Target: 1 if Player 1 wins
data["target"] = (data["Winner"] == data["Player_1"]).astype(int)

data["rank_diff"] = data["Rank_1"] - data["Rank_2"]
data["points_diff"] = data["Pts_1"] - data["Pts_2"]


# -----------------------------
# Initialise histories
# -----------------------------

player_history = {}
surface_history = {}
head_to_head = {}

overall_elo = {}
surface_elo = {}

recent_form_1 = []
recent_form_2 = []
surface_form_1 = []
surface_form_2 = []
h2h_diff = []

elo_diff = []
surface_elo_diff = []

FORM_MATCHES = 10
INITIAL_ELO = 1500
K = 20


def get_elo(ratings, player):
    return ratings.get(player, INITIAL_ELO)


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


# -----------------------------
# Build features chronologically
# -----------------------------

for _, row in data.iterrows():

    p1 = row["Player_1"]
    p2 = row["Player_2"]
    surface = row["Surface"]

    # ---------- Recent form ----------
    p1_history = player_history.get(p1, [])
    p2_history = player_history.get(p2, [])

    p1_form = (
        sum(p1_history[-FORM_MATCHES:]) /
        min(len(p1_history), FORM_MATCHES)
        if p1_history else 0.5
    )

    p2_form = (
        sum(p2_history[-FORM_MATCHES:]) /
        min(len(p2_history), FORM_MATCHES)
        if p2_history else 0.5
    )

    recent_form_1.append(p1_form)
    recent_form_2.append(p2_form)

    # ---------- Surface form ----------
    p1_surface_history = surface_history.get((p1, surface), [])
    p2_surface_history = surface_history.get((p2, surface), [])

    p1_surface_form = (
        sum(p1_surface_history[-FORM_MATCHES:]) /
        min(len(p1_surface_history), FORM_MATCHES)
        if p1_surface_history else 0.5
    )

    p2_surface_form = (
        sum(p2_surface_history[-FORM_MATCHES:]) /
        min(len(p2_surface_history), FORM_MATCHES)
        if p2_surface_history else 0.5
    )

    surface_form_1.append(p1_surface_form)
    surface_form_2.append(p2_surface_form)

    # ---------- Head-to-head ----------
    pair = tuple(sorted([p1, p2]))

    previous_meetings = head_to_head.get(
        pair,
        {"p1_wins": 0, "p2_wins": 0}
    )

    total_h2h = (
        previous_meetings["p1_wins"] +
        previous_meetings["p2_wins"]
    )

    if total_h2h == 0:
        h2h_difference = 0
    else:
        if pair[0] == p1:
            p1_h2h_wins = previous_meetings["p1_wins"]
            p2_h2h_wins = previous_meetings["p2_wins"]
        else:
            p1_h2h_wins = previous_meetings["p2_wins"]
            p2_h2h_wins = previous_meetings["p1_wins"]

        h2h_difference = (
            p1_h2h_wins / total_h2h -
            p2_h2h_wins / total_h2h
        )

    h2h_diff.append(h2h_difference)

    # ---------- Elo ----------
    p1_elo = get_elo(overall_elo, p1)
    p2_elo = get_elo(overall_elo, p2)

    elo_diff.append(p1_elo - p2_elo)

    # ---------- Surface Elo ----------
    p1_surface_elo = get_elo(
        surface_elo,
        (p1, surface)
    )

    p2_surface_elo = get_elo(
        surface_elo,
        (p2, surface)
    )

    surface_elo_diff.append(
        p1_surface_elo - p2_surface_elo
    )

    # ---------- Current result ----------
    p1_won = int(row["Winner"] == p1)
    p2_won = int(row["Winner"] == p2)

    # Update overall Elo AFTER prediction features are calculated
    expected_p1 = expected_score(p1_elo, p2_elo)

    overall_elo[p1] = (
        p1_elo + K * (p1_won - expected_p1)
    )

    overall_elo[p2] = (
        p2_elo + K * (p2_won - (1 - expected_p1))
    )

    # Update surface Elo
    expected_surface_p1 = expected_score(
        p1_surface_elo,
        p2_surface_elo
    )

    surface_elo[(p1, surface)] = (
        p1_surface_elo +
        K * (p1_won - expected_surface_p1)
    )

    surface_elo[(p2, surface)] = (
        p2_surface_elo +
        K * (p2_won - (1 - expected_surface_p1))
    )

    # Update form histories
    player_history.setdefault(p1, []).append(p1_won)
    player_history.setdefault(p2, []).append(p2_won)

    surface_history.setdefault(
        (p1, surface), []
    ).append(p1_won)

    surface_history.setdefault(
        (p2, surface), []
    ).append(p2_won)

    # Update H2H
    if pair not in head_to_head:
        head_to_head[pair] = {
            "p1_wins": 0,
            "p2_wins": 0
        }

    if p1 == pair[0]:
        head_to_head[pair]["p1_wins"] += p1_won
        head_to_head[pair]["p2_wins"] += p2_won
    else:
        head_to_head[pair]["p1_wins"] += p2_won
        head_to_head[pair]["p2_wins"] += p1_won


# -----------------------------
# Add features to dataframe
# -----------------------------

data["form_diff"] = (
    pd.Series(recent_form_1) -
    pd.Series(recent_form_2)
)

data["surface_form_diff"] = (
    pd.Series(surface_form_1) -
    pd.Series(surface_form_2)
)

data["h2h_diff"] = h2h_diff

data["elo_diff"] = elo_diff
data["surface_elo_diff"] = surface_elo_diff


# -----------------------------
# Train/test data
# -----------------------------

features = [
    "rank_diff",
    "points_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_diff",
    "elo_diff",
    "surface_elo_diff"
]

model_data = data[
    features + ["target"]
].dropna()

# Chronological split
split = int(len(model_data) * 0.8)

train = model_data.iloc[:split]
test = model_data.iloc[split:]

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]


# -----------------------------
# Train Logistic Regression
# -----------------------------

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


# -----------------------------
# Train Random Forest
# -----------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

import joblib

joblib.dump(random_forest, "tennis_model.pkl")


# -----------------------------
# Results
# -----------------------------

print(f"Logistic Regression: {logistic_accuracy * 100:.2f}%")
print(f"Random Forest:       {rf_accuracy * 100:.2f}%")
