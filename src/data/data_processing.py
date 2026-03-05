"""Data Processing Functions and Logic"""
from pathlib import Path
import pandas as pd
from pandas import DataFrame

from .utils import lazy


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

    def __init__(self, year: int = 2025, men: bool = True, refresh: bool = False) -> None:
        """
        Initializes `DataLoader` for a certain data pull year and gender

        Args:
            year: Year of the data pull
            men: boolean on if you want Mens (`True`) or Women's (`False`)
            refresh: if `True`, re-run preprocessing even if an interim file exists

        Returns:
            `None`, but initilaizes dataloader object
        """
        self.year = year
        self.gender = "M" if men else "W"
        self.refresh = refresh
        self.project_root = Path(__file__).resolve().parent.parent.parent

    @property
    def _interim_path(self) -> Path:
        """Path to the interim cached CSV for this year/gender."""
        return (
            self.project_root
            / "data"
            / "interim"
            / f"{self.gender}{self.year}_formatted_data.csv"
        )

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
            self._merge_coaches,
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

    def _merge_coaches(self, data: DataFrame) -> DataFrame:
        """Merge Coaches data on"""
        for team in ["team", "opponent"]:
            data = data.merge(
                self.coaches,
                left_on=["Season", f"{team}_TeamID"],
                right_on=["Season", "TeamID"],
                how="left",
            )

            # Filter where DayNum falls within the range
            data = data[
                (data["DayNum"] >= data["FirstDayNum"])
                & (data["DayNum"] <= data["LastDayNum"])
            ]

            data = data.rename(columns={"CoachName": f"{team}_CoachName"})
            data = data.drop(columns=["FirstDayNum", "LastDayNum", "TeamID"])

        return data

    @lazy
    def processed_data(self):
        """processed and formatted games data.

        Loads from the interim CSV if it exists and `refresh=False`.
        Otherwise runs the full preprocessing pipeline and writes the result
        to the interim file for future use.
        """
        if not self.refresh and self._interim_path.exists():
            print(f"Loading cached interim data from: {self._interim_path}")
            return pd.read_csv(self._interim_path)

        data = self.preprocess()
        self._write(data)
        return data

    @property
    def regular_season_data(self) -> DataFrame:
        """processed and formatted regular season games data."""
        return self.processed_data.loc[~self.processed_data["Tournament"]].copy()

    @property
    def tournament_data(self) -> DataFrame:
        """processed and formatted tournament games data."""
        return self.processed_data.loc[self.processed_data["Tournament"]].copy()

    def _write(self, data: DataFrame) -> None:
        """Write a DataFrame to the interim file."""
        self._interim_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self._interim_path, index=False)
        print(f"Wrote data to: {self._interim_path}")

    def write_processed_data(self) -> None:
        """Explicitly write processed data to the interim file."""
        self._write(self.processed_data)
