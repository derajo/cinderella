"""Data Processing Functions and Logic"""
from pathlib import Path
from typing import Callable
import pandas as pd
from pandas import DataFrame


def lazy(func: Callable) -> property:
    """Decorator for caching properties after first load."""
    attr_name = f"_{func.__name__}"

    @property
    def _lazy(self):
        """Gets attribute if it exists otherwise it sets the attribute"""
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy


class DataLoader:
    """
    `DataLoader` and basic formatter for NCAA M/W Basketball data
    """

    # File name constants
    REGULAR_SEASON_STATS_FILE = "RegularSeasonDetailedResults"
    TOURNAMENT_STATS_FILE = "NCAATourneyDetailedResults"
    TEAMS_FILE = "Teams"
    COACHES_FILE = "TeamCoaches"
    CONFERENCES_FILE = "TeamConferences"
    TOURNEY_SEEDS_FILE = "NCAATourneySeeds"
    TOURNEY_SLOTS_FILE = "NCAATourneySlots"

    def __init__(self, year: int = 2025, men: bool = True) -> None:
        """
        Initializes `DataLoader` for a certain data pull year and gender

        Args:
            year: Year of the data pull
            men: boolean on if you want Mens (`True`) or Women's (`False`)

        Returns:
            `None`, but initilaizes dataloader object
        """
        self.year = year
        self.gender = "M" if men else "W"
        self.project_root = Path().resolve().parent.parent

    def _load(self, file_constant: str) -> DataFrame:
        """Internal helper to load a CSV file by name constant."""
        path = (
            self.project_root
            / "data"
            / "raw"
            / f"march-machine-learning-mania-{self.year}"
            / f"{self.gender}{file_constant}.csv"
        )
        return pd.read_csv(path)

    # -----------------------------
    # Game-level data
    # -----------------------------
    @lazy
    def data(self):
        """load and concatenate regular and postseason data together."""
        # always concatenated from raw
        reg = self._load(self.REGULAR_SEASON_STATS_FILE)
        reg["Tournament"] = False

        tour = self._load(self.TOURNAMENT_STATS_FILE)
        tour["Tournament"] = True

        return pd.concat([reg, tour], ignore_index=True)

    # -----------------------------
    # Metadata
    # -----------------------------
    @lazy
    def teams(self):
        """Teams data such as IDs and Team names."""
        return self._load(self.TEAMS_FILE)

    @lazy
    def coaches(self):
        """Coaches data coach names and teams they coached for."""
        return self._load(self.COACHES_FILE)

    @lazy
    def conferences(self):
        """Conference information for teams"""
        return self._load(self.CONFERENCES_FILE)

    # -----------------------------
    # Tournament structure
    # -----------------------------
    @lazy
    def tourney_seeds(self):
        """Tournament seeding by year"""
        return self._load(self.TOURNEY_SEEDS_FILE)

    @lazy
    def tourney_slots(self):
        """Tournament slot information by year"""
        return self._load(self.TOURNEY_SLOTS_FILE)

    # -----------------------------
    # Preprocessing Pipeline
    # -----------------------------
    def preprocess(self) -> DataFrame:
        """Run all preprocessing steps and cache processed data."""
        data = self.data.copy()

        # Steps:
        steps = [
            self._create_game_id,
            self._possessions,
            self._minutes,
            self._create_team_game_stats,
            self._duplicate_and_flip_teams,
        ]

        for step in steps:
            print(step.__name__)
            data = step(data)

        return data

    # -----------------------------
    # Preprocessing Functions
    # -----------------------------

    def _possessions(self, data: DataFrame) -> DataFrame:
        """
        field goals attempted - offensive rebounds + turnovers + (0.475 x free throws attempted)
        """
        winning_team_poss = (
            data["WFGA"] - data["WOR"] + data["WTO"] + (0.475 * data["WFTA"])
        )
        losing_team_poss = (
            data["LFGA"] - data["LOR"] + data["LTO"] + (0.475 * data["LFTA"])
        )
        data["Possessions"] = ((winning_team_poss + losing_team_poss) // 2).astype(int)
        return data

    def _minutes(self, data: DataFrame) -> DataFrame:
        """calculate minutes in the game"""
        data["Minutes"] = 40 + data["NumOT"] * 5
        return data

    def _create_game_id(self, data: DataFrame) -> DataFrame:
        """GameID from season, DayNum, winning TeamID and Losing TeamID"""
        data["GameID"] = (
            data.Season.astype("str")
            + "_"
            + data.DayNum.astype(str).str.zfill(3)
            + "_"
            + data.WTeamID.astype("str")
            + "_"
            + data.LTeamID.astype("str")
        )
        return data

    def _create_team_game_stats(self, data: DataFrame) -> DataFrame:
        """rename columns from W/L to team/opponent."""
        data = data.rename(columns={"WLoc": "Loc"})
        winning_cols_mapping = {
            col: f"team_{col[1:]}" for col in data.columns if col.startswith("W")
        }
        losing_cols_mapping = {
            col: f"opponent_{col[1:]}"
            for col in data.columns
            if (col.startswith("L")) and (col != "Loc")
        }
        data = data.rename(columns=winning_cols_mapping)
        data = data.rename(columns=losing_cols_mapping)
        return data

    def _loser_loc_change(self, loc: str) -> str:
        """Changes the location to the opposite."""
        if loc == "H":
            return "A"
        if loc == "A":
            return "H"
        return "N"

    def _duplicate_and_flip_teams(self, data: DataFrame) -> DataFrame:
        """flip team and opponent and concatenate to original."""
        data_flipped = data.copy()
        data_flipped = data_flipped.rename(
            columns={
                col: col.replace("team_", "opponent_")
                if col.startswith("team_")
                else col.replace("opponent_", "team_")
                for col in data_flipped.columns
                if (col.startswith("team_")) | (col.startswith("opponent_"))
            }
        )

        data_flipped.Loc = data_flipped.Loc.apply(self._loser_loc_change)
        return pd.concat([data, data_flipped], ignore_index=True)

    @lazy
    def processed_data(self):
        """processed and formatted games data."""
        return self.preprocess()

    @property
    def regular_season_data(self) -> DataFrame:
        """processed and formatted regular season games data."""
        return self.processed_data.loc[~self.processed_data["Tournament"]].copy()

    @property
    def tournament_data(self) -> DataFrame:
        """processed and formatted tournament games data."""
        return self.processed_data.loc[self.processed_data["Tournament"]].copy()

    def write_processed_data(self) -> None:
        """pWrite data to interim table"""
        path = (
            self.project_root
            / "data"
            / "interim"
            / f"{self.gender}{self.year}_formatted_data.csv"
        )
        self.processed_data.to_csv(path, index=False)
        print(f"Wrote data to: {path}")
