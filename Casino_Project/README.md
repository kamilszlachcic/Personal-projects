# Casino Players Dataset Generator & Customer Churn Prediction

## Overview

This project consists of two main components:

1.  **Casino Players Dataset Generator** – A script that generates a synthetic dataset of casino players, simulating various gambling behaviors, session statistics, financial transactions, and churn indicators.
    
2.  **Customer Segmentation & Churn Prediction** – A machine learning model that analyzes player data to predict churn and segment customers based on their behavior.
    

## Part 1: Casino Players Dataset Generator

The dataset includes multiple features that describe the behavior and financial activity of players. Below is an explanation of how each feature is generated and why specific parameters were chosen.

### Features & Calculations

#### **1. Player_ID** (Integer)

-   Unique identifier for each player.
    

#### **2. Num_Sessions** (Integer)

-   Represents the number of gaming sessions a player has participated in.
    
-   Modeled using a Poisson distribution centered around **20**, as real casino players typically follow such a pattern.
    

#### **3. Avg_Session_Time** (Integer)

-   The average duration (in minutes) of a player's gaming session.
    
-   Assigned a random value between **5 to 180 minutes**, capturing both casual and hardcore players.
    

#### **4. Avg_Bet_Amount** (Integer)

-   The average bet amount per game session.
    
-   Selected from a predefined list **(5, 10, 20, 50, 100, 500, 1000)**, simulating real casino betting patterns.
    

#### **5. Num_Wins** (Integer)

-   The number of winning sessions.
    
-   **Players who place higher bets tend to win more games**, hence a bias is introduced for high rollers.
    

#### **6. Num_Losses** (Integer)

-   The number of losing sessions.
    
-   **Calculated as:**  `Num_Sessions - Num_Wins` to ensure logical consistency.
    

#### **7. Total_Winnings** (Float)

-   The total amount won across all sessions.
    
-   Computed as: `Num_Wins * Avg_Bet_Amount * random factor (0.8 to 2.0)` to introduce variability in winnings.
    

#### **8. Total_Losses** (Float)

-   The total amount lost across all sessions.
    
-   Computed as: `Num_Losses * Avg_Bet_Amount * random factor (0.5 to 1.5)`, adding variability to losses.
    

#### **9. Net_Profit** (Float)

-   The overall net profit or loss.
    
-   **Formula:**  `Total_Winnings - Total_Losses`.
    

#### **10. Favorite_Game** (Categorical)

-   Player's most frequently played casino game (e.g., "Slot Machines", "Poker", "Blackjack").
    
-   **Assigned based on probability distributions** reflecting real casino trends.
    

#### **11. Days_Since_Last_Play** (Integer)

-   Number of days since the player last participated in a session.
    
-   Randomly assigned between **0 to 60 days**, introducing player inactivity trends.
    

#### **12. Player_Type** (Categorical)

-   Categorizes players into different behavioral segments:
    
    -   **High Roller** – Large bets and frequent play.
        
    -   **Regular** – Consistent play with moderate bets.
        
    -   **Moderate** – Smaller bets but somewhat frequent activity.
        
    -   **Casual** – Low activity and low betting amounts.
        
-   **Calculated based on:**  `Avg_Bet_Amount` and `Num_Sessions` thresholds.
    

#### **13. Active_Days_Per_Month** (Integer)

-   The estimated number of active days per month.
    
-   **Derived from:**  `Num_Sessions / 2` with small random adjustments.
    

#### **14. Used_Bonuses** (Boolean)

-   Indicates if a player has used casino bonuses.
    
-   Randomly assigned **(50-50 probability)**.
    

#### **15. Total_Deposit** (Float)

-   The total money deposited by the player.
    
-   Randomly assigned between **$50 and $10,000**.
    

#### **16. Total_Withdrawal** (Float)

-   The total money withdrawn by the player.
    
-   **Computed as:**  `30%-90% of Total_Deposit`, mimicking real casino withdrawal behavior.
    

#### **17. Avg_Reaction_Time** (Float)

-   The player’s average reaction time in making decisions (in seconds).
    
-   **Generated between:**  `0.5 to 5.0 seconds`, with shorter reaction times reflecting experienced players.
    

#### **18. Trend_Sessions** (Float)

-   Measures session increase or decrease over time.
    
-   Computed as the difference from the previous session count.
    

#### **19. Frustration_Score** (Integer)

-   A score reflecting player frustration levels.
    
-   Based on:
    
    -   **High losses** (Net Profit < -60% of deposits)
        
    -   **Many losing sessions**
        
    -   **Lack of bonus usage**
        

#### **20. Churn** (Boolean)

-   Indicates whether the player has stopped playing.
    
-   **Determined based on:** long inactivity periods, high losses, or a downward session trend.
    

#### **21. Soft_Churn** (Boolean)

-   Indicates early signs of churn (not yet fully inactive).
    
-   Based on **moderate inactivity and decreasing play frequency**.
    

----------

## Part 2: Customer Segmentation & Churn Prediction

### **Modeling Approach**

We train five machine learning models to predict player churn:

-   **RandomForest**
    
-   **XGBoost**
    
-   **LightGBM**
    
-   **CatBoost**
    
-   **Logistic Regression**
    

### **Resampling Strategy**

**Before Resampling:**

-   `Churn 1: 48,818`
    
-   `Churn 0: 31,182`
    

**After Resampling:**

-   `Churn 1: 48,817`
    
-   `Churn 0: 34,172`
    

Since resampling barely changes the class balance, we might reconsider its necessity.

### **Threshold Selection**

-   We use a **threshold of 0.6** for all models, ensuring more reliable predictions.
    
-   Previously, lower thresholds (e.g., 0.4) led to excessive recall and F1-score close to 1, suggesting overfitting.
    

### **Post-Training Outputs**

-   After each model completes training, a message appears: `"Model {model_name} training completed"`
    
-   A **comparison table** of all models is generated with key metrics (accuracy, precision, recall, F1-score).
    
-   All trained models are saved with their respective names for later use.
    

----------

## Usage

### **Installing Dependencies**

Run the following command to install required packages:

```
pip install -r requirements.txt
```

### **Generating the Dataset**

Run the dataset generator script:

```
python Casino_Dataset_Generator.py
```

### **Training the Churn Prediction Model**

Run the customer segmentation and churn prediction script:

```
python Customer_Segmentation_Churn_Prediction.py
```

----------

## Summary

This project provides a comprehensive solution for casino player data analysis, including dataset generation, segmentation, and churn prediction. The structured data and machine learning models can help casinos identify at-risk players and optimize customer retention strategies.

## License

This project is open-source and can be used for research and educational purposes.
