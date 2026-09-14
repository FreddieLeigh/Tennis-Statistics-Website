import streamlit as st
import pandas as pd
import plotly.express as px

from model import (
    train_model,
    get_prediction_features,
    FEATURES
)

st.title("Tennis analytics")

data = pd.read_csv("data/tennis.csv")

@st.cache_resource
def load_model():
    model, logistic_accuracy, rf_accuracy, _ = train_model(data)
    return model, logistic_accuracy, rf_accuracy

model, logistic_accuracy, rf_accuracy = load_model()

@st.cache_data
def get_cached_prediction_features(player_1, player_2, surface):
    return get_prediction_features(
        data,
        player_1,
        player_2,
        surface
    )

st.write("Number of matches:", len(data))


players = sorted(set(data["Player_1"]).union(data["Player_2"]))

player = st.selectbox(
    "Select a Player:", players)

player_matches = data[
    (data["Player_1"] == player) |
    (data["Player_2"] == player)
].copy()


wins = (player_matches["Winner"] == player).sum()
matches_played = len(player_matches)
losses = matches_played - wins
win_rate = wins / matches_played * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Matches Played", matches_played)

with col2:
    st.metric("Wins", wins)

with col3:
    st.metric("Win Rate", f"{win_rate:.1f}%")

surface_stats = []

for surface in player_matches["Surface"].dropna().unique():
     surface_matches = player_matches[player_matches["Surface"] == surface]

     surface_wins = (surface_matches["Winner"] == player).sum()
     surface_total = len(surface_matches)

     surface_stats.append({
         "Surface": surface,
         "Win Rate": surface_wins / surface_total * 100
    })

surface_data = pd.DataFrame(surface_stats)

st.subheader(f"{player} - Win Rate by Surface")

fig = px.bar(
    surface_data,
    x="Surface",
    y="Win Rate",
    text="Win Rate"
)


fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(yaxis_title="Win Rate (%)", xaxis_title="")#maybe change
                  

st.plotly_chart(fig, use_container_width=True)

player_matches["Year"] = pd.to_datetime(
    player_matches["Date"]
).dt.year

yearly_stats = (
    player_matches
    .groupby("Year")
    .agg(
        Matches=("Winner", "size"),
        Wins=("Winner", lambda x: (x == player).sum())
    )
    .reset_index()
)

yearly_stats["Win Rate"] = (
    yearly_stats["Wins"] /
    yearly_stats["Matches"] * 100
)

st.subheader(f"{player} — Win Rate by Year")

fig = px.line(
    yearly_stats,
    x="Year",
    y="Win Rate",
    markers=True
)

fig.update_layout(
    yaxis_title="Win Rate (%)",
    xaxis_title="Year"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Head-to-Head")

opponents = sorted(set(
    player_matches["Player_1"].tolist() +
    player_matches["Player_2"].tolist()
))

opponents = [p for p in opponents if p != player]

opponent = st.selectbox(
    "Choose an opponent",
    opponents
)

h2h = player_matches[
    (player_matches["Player_1"] == opponent) |
    (player_matches["Player_2"] == opponent)
].copy()


surfaces = ["All"] + sorted (h2h["Surface"].dropna().unique().tolist())

selected_surface = st.selectbox(
    "Surface",
    surfaces
)
if selected_surface != "All":
    h2h_filtered = h2h[h2h["Surface"] == selected_surface]
else:
    h2h_filtered = h2h
    
player_wins = (h2h_filtered["Winner"] == player).sum()
opponent_wins = (h2h_filtered["Winner"] == opponent).sum()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"{player} Wins", player_wins)

with col2:
    st.metric(f"{opponent} Wins", opponent_wins)

with col3:
    st.metric("Matches", len(h2h))

st.subheader("Match History")

display_data = h2h_filtered[
    ["Date", "Tournament", "Surface", "Round", "Winner", "Score"]
    ].sort_values("Date", ascending = False)

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)

st.subheader("Match Prediction")

player_1 = st.selectbox(
    "Player 1",
    players,
    key="prediction_player_1"
)

player_2 = st.selectbox(
    "Player 2",
    [p for p in players if p != player_1],
    key="prediction_player_2"
)

prediction_surface = st.selectbox(
    "Surface",
    sorted(data["Surface"].dropna().unique()),
    key="prediction_surface"
)

prediction_features = get_cached_prediction_features(
    player_1,
    player_2,
    prediction_surface
)

probabilities = model.predict_proba(
    prediction_features
)[0]

probability_1 = probabilities[1]
probability_2 = probabilities[0]

if probability_1 >= probability_2:
    predicted_winner = player_1
    winner_probability = probability_1
else:
    predicted_winner = player_2
    winner_probability = probability_2


st.success(
    f" Predicted Winner: **{predicted_winner}** "
    f"({winner_probability * 100:.1f}% probability)"
)


col1, col2 = st.columns(2)

with col1:
    st.metric(
        player_1,
        f"{probability_1 * 100:.1f}%"
    )

with col2:
    st.metric(
        player_2,
        f"{probability_2 * 100:.1f}%"
    )


prediction_chart = pd.DataFrame({
    "Player": [player_1, player_2],
    "Win Probability": [
        probability_1 * 100,
        probability_2 * 100
    ]
})

fig = px.bar(
    prediction_chart,
    x="Player",
    y="Win Probability",
    range_y=[0, 100],
    title=f"Predicted Win Probability on {prediction_surface}"
)

fig.update_layout(
    yaxis_title="Probability (%)",
    xaxis_title="",
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Why does the model make this prediction?")

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

feature_importance["Feature"] = feature_importance["Feature"].replace({
    "rank_diff": "Ranking Difference",
    "points_diff": "Ranking Points Difference",
    "form_diff": "Recent Form",
    "surface_form_diff": "Surface Form",
    "h2h_diff": "Head-to-Head",
    "elo_diff": "Overall Elo",
    "surface_elo_diff": "Surface Elo"
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=True
)

importance_fig = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Random Forest Feature Importance"
)

importance_fig.update_layout(
    xaxis_title="Importance",
    yaxis_title=""
)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)
st.subheader("Prediction Inputs")

display_features = prediction_features.copy()

display_features.columns = [
    "Ranking Difference",
    "Ranking Points Difference",
    "Recent Form Difference",
    "Surface Form Difference",
    "Head-to-Head Difference",
    "Overall Elo Difference",
    "Surface Elo Difference"
]

st.dataframe(
    display_features,
    use_container_width=True
)

st.caption(
    "Predictions are based on historical data through the latest "
    "match available in the dataset."
)
