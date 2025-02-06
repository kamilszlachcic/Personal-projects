import pandas as pd
import numpy as np

# Set randomness for reproducibility
np.random.seed(42)

# Number of players in the simulation
num_players = 100000

# Generate player IDs
player_ids = np.arange(1, num_players + 1)

# Generate number of sessions per player
num_sessions = np.random.poisson(lam=20, size=num_players)

# Generate average session time
avg_session_time = np.random.randint(5, 180, size=num_players)

# Generate average bet amounts
avg_bet_amount = np.random.choice([5, 10, 20, 50, 100, 500, 1000], size=num_players,
                                  p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05])

# Generate number of wins and losses per session
num_wins = np.where(avg_bet_amount >= 500,
    np.random.randint(5, np.maximum(num_sessions, 6), size=num_players),
    np.random.randint(0, np.maximum(num_sessions // 2, 1), size=num_players))
num_losses = num_sessions - num_wins

# Total winnings and losses
total_winnings = num_wins * avg_bet_amount * np.random.uniform(0.8, 2.0, size=num_players)
total_losses = num_losses * avg_bet_amount * np.random.uniform(0.5, 1.5, size=num_players)
net_profit = total_losses - total_winnings

# Favorite game
favorite_game = np.random.choice(["Slot Machines", "Blackjack", "Poker", "Roulette", "Sports Betting"],
                                 size=num_players, p=[0.4, 0.2, 0.2, 0.1, 0.1])

# Days since last play
days_since_last_play = np.random.randint(0, 60, size=num_players)

# Player categories
def categorize_player(bet_amount, num_sessions):
    if bet_amount > 1000 or num_sessions > 50:
        return "High Roller"
    elif bet_amount >= 50 and num_sessions > 20:
        return "Regular"
    elif num_sessions < 5:
        return "Casual"
    else:
        return "Moderate"

player_type = np.array([categorize_player(bet, sessions) for bet, sessions in zip(avg_bet_amount, num_sessions)])

# Active days per month
active_days_per_month = np.clip(np.random.normal(loc=num_sessions // 2, scale=3, size=num_players), 1, 30).astype(int)

# Whether the player used bonuses
used_bonuses = np.random.choice([0, 1], size=num_players, p=[0.5, 0.5])

# Deposits and withdrawals
total_deposit = np.round(np.random.uniform(50, 10000, size=num_players), 2)
total_withdrawal = np.round(total_deposit * np.random.uniform(0.3, 0.9, size=num_players), 2)

# Reaction time
total_reaction_time = np.random.uniform(0.5, 5.0, size=num_players)

# Player segmentation
casino_data = pd.DataFrame({
    "Player_ID": player_ids,
    "Num_Sessions": num_sessions,
    "Avg_Session_Time": avg_session_time,
    "Avg_Bet_Amount": avg_bet_amount,
    "Num_Wins": num_wins,
    "Num_Losses": num_losses,
    "Total_Winnings": total_winnings,
    "Total_Losses": total_losses,
    "Net_Profit": net_profit,
    "Favorite_Game": favorite_game,
    "Days_Since_Last_Play": days_since_last_play,
    "Player_Type": player_type,
    "Active_Days_Per_Month": active_days_per_month,
    "Used_Bonuses": used_bonuses,
    "Total_Deposit": total_deposit,
    "Total_Withdrawal": total_withdrawal,
    "Avg_Reaction_Time": total_reaction_time
})

# Calculate Session Trend
casino_data["Trend_Sessions"] = casino_data["Num_Sessions"].diff().fillna(0)

# Frustration Score (random influence on churn)
casino_data["Frustration_Score"] = (
    (casino_data["Net_Profit"] < -0.6 * casino_data["Total_Deposit"]).astype(int) +
    (casino_data["Num_Losses"] > np.random.randint(3, 8)).astype(int) +
    (casino_data["Used_Bonuses"] == 0).astype(int)
)

# Churn logic
casino_data["Churn"] = np.where(
    (casino_data["Days_Since_Last_Play"] > np.random.randint(15, 45)) |
    (casino_data["Net_Profit"] < -0.5 * casino_data["Total_Deposit"]) |
    ((casino_data["Num_Sessions"] < 10) & (casino_data["Used_Bonuses"] == 0)) |
    ((casino_data["Trend_Sessions"] < -3) & (casino_data["Net_Profit"] < -0.3 * casino_data["Total_Deposit"])) &
    (np.random.rand(num_players) < 0.7),
    1, 0
)

# Soft Churn logic
casino_data["Soft_Churn"] = np.where(
    (casino_data["Days_Since_Last_Play"] > 20) & (casino_data["Days_Since_Last_Play"] < 40) &
    (casino_data["Trend_Sessions"] < -0.3 * casino_data["Num_Sessions"]) &
    (casino_data["Net_Profit"] > -0.5 * casino_data["Total_Deposit"]) &
    (casino_data["Player_Type"] != "High Roller"),
    1, 0
)

# Save data to CSV file
casino_data.to_csv("casino_players_data.csv", index=False)
print("Dataset generated and saved as 'casino_players_data.csv'")