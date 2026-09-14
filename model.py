import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


FORM_MATCHES = 10
INITIAL_ELO = 1500
K = 20

FEATURES = [
    "rank_diff",
    "points_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_diff",
    "elo_diff",
    "surface_elo_diff"
]


def get_elo(ratings, player):
    return ratings.get(player, INITIAL_ELO)


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def build_features(data):
    """Build leakage-free historical features for every match."""

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data["target"] = (
        data["Winner"] == data["Player_1"]
    ).astype(int)

    data["rank_diff"] = (
        data["Rank_1"] - data["Rank_2"]
    )

    data["points_diff"] = (
        data["Pts_1"] - data["Pts_2"]
    )

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

    for _, row in data.iterrows():

        p1 = row["Player_1"]
        p2 = row["Player_2"]
        surface = row["Surface"]

        # -------------------------
        # Recent form
        # -------------------------

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

        # -------------------------
        # Surface form
        # -------------------------

        p1_surface_history = surface_history.get(
            (p1, surface), []
        )

        p2_surface_history = surface_history.get(
            (p2, surface), []
        )

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

        # -------------------------
        # Head-to-head
        # -------------------------

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

        # -------------------------
        # Overall Elo
        # -------------------------

        p1_elo = get_elo(overall_elo, p1)
        p2_elo = get_elo(overall_elo, p2)

        elo_diff.append(p1_elo - p2_elo)

        # -------------------------
        # Surface Elo
        # -------------------------

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

        # -------------------------
        # Match result
        # -------------------------

        p1_won = int(row["Winner"] == p1)
        p2_won = int(row["Winner"] == p2)

        # Update Elo AFTER features are calculated
        expected_p1 = expected_score(
            p1_elo,
            p2_elo
        )

        overall_elo[p1] = (
            p1_elo +
            K * (p1_won - expected_p1)
        )

        overall_elo[p2] = (
            p2_elo +
            K * (p2_won - (1 - expected_p1))
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
            K * (
                p2_won -
                (1 - expected_surface_p1)
            )
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

    # -------------------------
    # Add engineered features
    # -------------------------

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

    return data


def train_model(data):
    """Train the Random Forest and return it with test accuracy."""

    featured_data = build_features(data)

    model_data = featured_data[
        FEATURES + ["target"]
    ].dropna()

    # Chronological 80/20 split
    split = int(len(model_data) * 0.8)

    train = model_data.iloc[:split]
    test = model_data.iloc[split:]

    X_train = train[FEATURES]
    y_train = train["target"]

    X_test = test[FEATURES]
    y_test = test["target"]

    # Logistic Regression baseline
    logistic_model = LogisticRegression(
        max_iter=1000
    )

    logistic_model.fit(X_train, y_train)

    logistic_predictions = (
        logistic_model.predict(X_test)
    )

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions
    )

    # Random Forest
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

    return (
        random_forest,
        logistic_accuracy,
        rf_accuracy,
        featured_data
    )

def get_prediction_features(data, player_1, player_2, surface):
    """
    Calculate the seven model features for a future matchup.
    Uses all historical matches in the dataset.
    """

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    player_history = {}
    surface_history = {}
    head_to_head = {}

    overall_elo = {}
    surface_elo = {}

    for _, row in data.iterrows():

        p1 = row["Player_1"]
        p2 = row["Player_2"]
        match_surface = row["Surface"]

        p1_won = int(row["Winner"] == p1)
        p2_won = int(row["Winner"] == p2)

        # Current Elo BEFORE this match
        p1_elo = get_elo(overall_elo, p1)
        p2_elo = get_elo(overall_elo, p2)

        p1_surface_elo = get_elo(
            surface_elo,
            (p1, match_surface)
        )

        p2_surface_elo = get_elo(
            surface_elo,
            (p2, match_surface)
        )

        # Update overall Elo
        expected_p1 = expected_score(
            p1_elo,
            p2_elo
        )

        overall_elo[p1] = (
            p1_elo +
            K * (p1_won - expected_p1)
        )

        overall_elo[p2] = (
            p2_elo +
            K * (p2_won - (1 - expected_p1))
        )

        # Update surface Elo
        expected_surface_p1 = expected_score(
            p1_surface_elo,
            p2_surface_elo
        )

        surface_elo[(p1, match_surface)] = (
            p1_surface_elo +
            K * (
                p1_won -
                expected_surface_p1
            )
        )

        surface_elo[(p2, match_surface)] = (
            p2_surface_elo +
            K * (
                p2_won -
                (1 - expected_surface_p1)
            )
        )

        # Update histories
        player_history.setdefault(p1, []).append(p1_won)
        player_history.setdefault(p2, []).append(p2_won)

        surface_history.setdefault(
            (p1, match_surface), []
        ).append(p1_won)

        surface_history.setdefault(
            (p2, match_surface), []
        ).append(p2_won)

        # Update H2H
        pair = tuple(sorted([p1, p2]))

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

    # --------------------------------
    # Ranking and points
    # --------------------------------

    def get_latest_rank_points(player):
        matches = data[
            (data["Player_1"] == player) |
            (data["Player_2"] == player)
        ]

        latest = matches.iloc[-1]

        if latest["Player_1"] == player:
            return latest["Rank_1"], latest["Pts_1"]

        return latest["Rank_2"], latest["Pts_2"]

    rank_1, points_1 = get_latest_rank_points(player_1)
    rank_2, points_2 = get_latest_rank_points(player_2)

    # --------------------------------
    # Recent form
    # --------------------------------

    def get_form(player):
        history = player_history.get(player, [])

        if not history:
            return 0.5

        recent = history[-FORM_MATCHES:]

        return sum(recent) / len(recent)

    form_1 = get_form(player_1)
    form_2 = get_form(player_2)

    # --------------------------------
    # Surface form
    # --------------------------------

    def get_surface_form(player):
        history = surface_history.get(
            (player, surface),
            []
        )

        if not history:
            return 0.5

        recent = history[-FORM_MATCHES:]

        return sum(recent) / len(recent)

    surface_form_1 = get_surface_form(player_1)
    surface_form_2 = get_surface_form(player_2)

    # --------------------------------
    # Head-to-head
    # --------------------------------

    pair = tuple(sorted([player_1, player_2]))

    meetings = head_to_head.get(
        pair,
        {"p1_wins": 0, "p2_wins": 0}
    )

    total_h2h = (
        meetings["p1_wins"] +
        meetings["p2_wins"]
    )

    if total_h2h == 0:

        h2h_difference = 0

    else:

        if pair[0] == player_1:

            p1_h2h = meetings["p1_wins"]
            p2_h2h = meetings["p2_wins"]

        else:

            p1_h2h = meetings["p2_wins"]
            p2_h2h = meetings["p1_wins"]

        h2h_difference = (
            p1_h2h / total_h2h -
            p2_h2h / total_h2h
        )

    # --------------------------------
    # Elo
    # --------------------------------

    p1_elo = get_elo(overall_elo, player_1)
    p2_elo = get_elo(overall_elo, player_2)

    p1_surface_elo = get_elo(
        surface_elo,
        (player_1, surface)
    )

    p2_surface_elo = get_elo(
        surface_elo,
        (player_2, surface)
    )

    # --------------------------------
    # Final feature vector
    # --------------------------------

    return pd.DataFrame([{
        "rank_diff": rank_1 - rank_2,
        "points_diff": points_1 - points_2,
        "form_diff": form_1 - form_2,
        "surface_form_diff": (
            surface_form_1 -
            surface_form_2
        ),
        "h2h_diff": h2h_difference,
        "elo_diff": p1_elo - p2_elo,
        "surface_elo_diff": (
            p1_surface_elo -
            p2_surface_elo
        )
    }])[FEATURES]

if __name__ == "__main__":

    data = pd.read_csv("data/tennis.csv")

    model, logistic_accuracy, rf_accuracy, featured_data = (
        train_model(data)
    )

    joblib.dump(model, "tennis_model.pkl")

    print(
        f"Logistic Regression: "
        f"{logistic_accuracy * 100:.2f}%"
    )

    print(
        f"Random Forest: "
        f"{rf_accuracy * 100:.2f}%"
    )
