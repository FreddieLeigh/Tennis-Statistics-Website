import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Tennis analytics")

data = pd.read_csv("data/tennis.csv")

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

p1_matches = data[
    ((data["Player_1"] == player_1) | (data["Player_2"] == player_1)) &
    (data["Surface"] == prediction_surface)
]

p2_matches = data[
    ((data["Player_1"] == player_2) | (data["Player_2"] == player_2)) &
    (data["Surface"] == prediction_surface)
]

p1_win_rate = (p1_matches["Winner"] == player_1).mean()
p2_win_rate = (p2_matches["Winner"] == player_2).mean()

latest_p1 = data[
    (data["Player_1"] == player_1) | (data["Player_2"] == player_1)
].sort_values("Date").iloc[-1]

latest_p2 = data[
    (data["Player_1"] == player_2) | (data["Player_2"] == player_2)
].sort_values("Date").iloc[-1]

if latest_p1["Player_1"] == player_1:
    rank_1 = latest_p1["Rank_1"]
else:
    rank_1 = latest_p1["Rank_2"]

if latest_p2["Player_1"] == player_2:
    rank_2 = latest_p2["Rank_1"]
else:
    rank_2 = latest_p2["Rank_2"]


score_1 = (1 / rank_1) * 0.4 + p1_win_rate * 0.6
score_2 = (1 / rank_2) * 0.4 + p2_win_rate * 0.6

probability_1 = score_1 / (score_1 + score_2)
probability_2 = 1 - probability_1

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

