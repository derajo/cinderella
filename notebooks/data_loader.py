# %% Setup
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processing import DataLoader

# %% Basic instantiation
loader = DataLoader(year=2025, men=True)
print(f"project_root : {loader.project_root}")
print(f"interim path : {loader._interim_path}")
print(f"exists       : {loader._interim_path.exists()}")

# %% First access: runs preprocessing pipeline + writes interim file
# (or loads from interim if already present)
loader.processed_data.head()

# %% Cache hit: new instance reads from interim CSV instead of reprocessing
loader2 = DataLoader(year=2025, men=True)
loader2.processed_data.head()

# %% refresh=True: ignores interim file and re-runs the full pipeline
loader_refresh = DataLoader(year=2025, men=True, refresh=True)
loader_refresh.processed_data.shape

# %% Regular season vs tournament split
print("Regular season rows:", len(loader.regular_season_data))
print("Tournament rows    :", len(loader.tournament_data))
loader.tournament_data[["Season", "GameID", "team_TeamID", "team_Score", "opponent_TeamID", "opponent_Score"]].tail()

# %% Metadata: teams, coaches, conferences
display(loader.teams.head(3))
display(loader.coaches.head(3))
display(loader.conferences.head(3))

# %% Tournament structure: seeds and slots
display(loader.tourney_seeds.query("Season == 2025").head(8))
display(loader.tourney_slots.query("Season == 2025").head(8))

# %% Women's data (men=False)
w_loader = DataLoader(year=2025, men=False)
print(f"interim path : {w_loader._interim_path}")
w_loader.processed_data.shape
