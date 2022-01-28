from __future__ import annotations

import datetime
import gc
import os
import random
import subprocess
from pickle import UnpicklingError
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


class DataGenerator:
    """Data generator"""

    file_name: str = "test_data"
    batch_size: int = 1000
    columns_count: int = 2
    file_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    batch_types: List[object] = [int, str, list, datetime.datetime]
    list_size: int = 3

    def __init__(
        self,
        file_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        columns_count: Optional[int] = None,
        file_size: Optional[int] = None,
        allow_empty: bool = True,
        batch_types: Optional[List[object]] = None,
        list_size: Optional[int] = None,
    ) -> None:
        """Initial data generator"""
        self.file_name = file_name or self.file_name
        self.batch_size = batch_size or self.batch_size
        self.columns_count = columns_count or self.columns_count
        self.file_size = file_size or self.file_size
        self.allow_empty: bool = allow_empty
        self.batch_types = batch_types or self.batch_types
        self.list_size = list_size or self.list_size
        if os.path.exists(f"./{self.file_name}.pkl"):
            self.df = pd.read_pickle(f"./{self.file_name}.pkl")
        else:
            self.dtypes: Dict[str, object] = {
                chr(i + 97): random.choice(self.batch_types)
                for i in range(self.columns_count)
            }
            self.df: DataFrame = self.create_df()

    def add_empty(self, batch: Iterable[Any]) -> Generator[Any]:
        """Add None to batch generator"""
        if self.allow_empty:
            return batch
        return (random.choices((v, None), weights=(95, 5)) for v in batch)

    def generate_batch_int(self, batch_size: int) -> Generator[Optional[int]]:
        """Generate int batch"""
        randrange: Tuple[int, int] = (1, self.batch_size)
        if hasattr(self, "df"):
            df_count: int = len(self.df)
            randrange = (
                1 + df_count,
                self.batch_size + df_count,
            )  # create range numbers
        return (random.randint(*randrange) for _ in range(batch_size))

    @staticmethod
    def generate_batch_str(batch_size: int) -> Iterable[Optional[str]]:
        """Generate str batch"""
        return map(
            chr,
            (
                random.randint(65, 90)
                if bool(random.getrandbits(1))
                else random.randint(97, 122)
                for _ in range(batch_size)
            ),
        )

    @staticmethod
    def generate_batch_datetime(
        batch_size: int,
    ) -> Iterable[Optional[datetime.datetime]]:
        """Generate datetime batch"""
        now: datetime.datetime = datetime.datetime.now()
        return map(
            datetime.datetime.fromtimestamp,
            (random.randint(1, int(now.timestamp())) for _ in range(batch_size)),
        )

    def generate_batch_list(
        self, batch_size: int
    ) -> Iterable[Optional[List[Optional[str]]]]:
        """Generate list str batch"""
        return map(
            list,
            (
                (random.randint(1, self.batch_size) for _ in range(self.list_size))
                for _ in range(batch_size)
            ),
        )

    def generate_batch(
        self, batch_type: Optional[object] = None, batch_size: Optional[int] = None
    ) -> Generator[Any]:
        """Generate bach"""
        if batch_type is None:
            batch_type: object = random.choice(self.batch_types)
        batch_size: int = batch_size or self.batch_size
        batch: Iterable[Any] = getattr(self, f"generate_batch_{batch_type.__name__}")(
            batch_size=batch_size
        )
        return self.add_empty(batch=batch)

    def create_df(self) -> DataFrame:
        """Create new df"""
        dtypes: Dict[object, str] = {
            int: "int64",
            str: "str",
            datetime.datetime: "datetime64[ns]",
            list: "object",
        }
        return pd.DataFrame(
            columns={
                col: pd.Series(dtype=dtypes.get(d_type, "str"))
                for col, d_type in self.dtypes.items()
            }
        )

    def update_df(self) -> DataFrame:
        """Add rows to DataFrame"""
        return self.df.append(
            pd.DataFrame(
                {
                    col: self.generate_batch(batch_type=d_type)
                    for col, d_type in self.dtypes.items()
                }
            ),
            ignore_index=True,
        )

    def save_df(self, file_name: Optional[str] = None) -> None:
        """Save DataFrame to pickle file"""
        file_name = file_name or self.file_name
        self.df.to_pickle(f"./{file_name}.pkl")

    def generate_df(self) -> None:
        """Generate new DataFrame and save"""

        self.save_df()

        with tqdm(total=self.file_size) as pbar:
            current_size: int = os.path.getsize(f"./{self.file_name}.pkl")
            while current_size < self.file_size:
                self.df = self.update_df()
                self.save_df()
                current_size = os.path.getsize(f"./{self.file_name}.pkl")
                pbar.update(current_size - pbar.n)
            else:
                self.df = self.df.iloc[: -self.batch_size]
                self.save_df()

    def create_copy(self, file_name: Optional[str] = None) -> DataGenerator:
        """Create equal DataFrame"""
        new_data_generator: DataGenerator = DataGenerator(
            file_name=self.file_name,
            batch_size=self.batch_size,
            columns_count=self.columns_count,
            file_size=self.file_size,
            allow_empty=self.allow_empty,
            batch_types=self.batch_types,
            list_size=self.list_size,
        )
        new_data_generator.df = self.df.copy(deep=True)
        new_data_generator.dtypes = self.dtypes
        new_data_generator.file_name = file_name or f"copy_{self.file_name}"
        return new_data_generator

    def add_column(self, batch_type: Optional[object] = None) -> None:
        """Add new column to DataFrame"""
        char: str = chr(len(self.df.columns) + 97)
        self.dtypes.update({char: random.choice(self.batch_types)})
        self.df[char] = tuple(
            self.generate_batch(
                batch_type=batch_type or self.dtypes[char], batch_size=len(self.df)
            )
        )

    def shuffle_columns(self) -> None:
        """Change columns order"""
        columns: List[str] = self.df.columns.to_list()
        random.shuffle(columns)
        self.df = self.df[columns]

    def shuffle_rows(self) -> None:
        """Change columns order"""
        self.df = self.df.sample(frac=1)

    def drop_rows(self, cnt: int = 100) -> None:
        """Drop random rows from DataSet"""
        self.df.drop(
            self.df.index[[random.randint(0, len(self.df)) for _ in range(cnt)]],
            inplace=True,
        )

    def change_data_in_column(
        self, column: Optional[str] = None, change_count: Optional[int] = None
    ) -> None:
        """Change data in selected column"""
        column = column or random.choice(tuple(self.dtypes.keys()))
        change_count = change_count or int(len(self.df) / 10) or 1
        with tqdm(total=change_count) as pbar:
            for _ in range(change_count):
                self.df.at[random.randint(1, len(self.df)), column] = next(
                    self.generate_batch(batch_type=self.dtypes[column], batch_size=1)
                )
                pbar.update(1)


class EqualDataFrames:
    """Equal two dataframes"""

    df1: DataFrame
    df2: DataFrame
    types: Dict

    def __init__(self, df1_path: str, df2_path: str) -> None:
        """Initial equals DataFrames"""
        self.df1_path: str = df1_path
        self.df2_path: str = df2_path

    def read_data_frames(self) -> bool:
        """Read pickle file"""
        if getattr(self, "df1", None) is None or getattr(self, "df2", None) is None:
            try:
                self.df1 = pd.read_pickle(self.df1_path)
                self.df2 = pd.read_pickle(self.df2_path)
            except (FileNotFoundError, OSError, UnpicklingError):
                return False
        return True

    @property
    def equal_columns(self) -> bool:
        """Equals columns count and order"""
        assert (
            self.df1 is not None or self.df2 is not None
        ), "Need read DataFrames before equals"
        return (
            len(self.df1.columns)
            and len(self.df1.columns) == len(self.df2.columns)
            and all(self.df1.columns == self.df2.columns)
        )

    @property
    def equal_rows_count(self) -> bool:
        """Equals rows count"""
        assert (
            self.df1 is not None or self.df2 is not None
        ), "Need read DataFrames before equals"
        return len(self.df1) == len(self.df2)

    @property
    def equal_columns_types(self) -> bool:
        """Equals columns types"""
        if dict(self.df1.dtypes) == dict(self.df2.dtypes):
            if len(self.df1) and len(self.df2):
                df_types1: Dict[str, str] = dict(self.df1.dtypes)
                df_types2: Dict[str, str] = dict(self.df1.dtypes)
                types: Dict[str, Dict] = {"df1": df_types1, "df2": df_types2}

                dtypes: Dict[str, object] = {
                    "int64": int,
                    "str": str,
                    "datetime64[ns]": datetime.datetime,
                }
                for df, df_type in types.items():
                    for col, tp in types[df].items():
                        if tp != "object":
                            types[df][col] = dtypes.get(types[df][col], "object")
                        else:
                            first_index_row: int = getattr(self, df)[
                                col
                            ].first_valid_index()
                            if first_index_row is not None:
                                types[df][col] = type(
                                    getattr(self, df)[col].loc[first_index_row]
                                )
                if types["df1"] == types["df2"]:
                    self.types = types
                    return True
        return False

    @property
    def equal_dataframes(self) -> bool:
        """Equals two DateFrames"""
        for df_title, types in self.types.items():
            for col in (col for col, tp in types.items() if tp == list):
                df = getattr(self, df_title)
                df[col] = df[col].apply(lambda x: x.sort() or tuple(x) if x else None)
        return self.df1.sort_values(by=self.df1.columns.to_list()).equals(
            self.df2.sort_values(by=self.df2.columns.to_list())
        )

    @property
    def equals(self) -> bool:
        """Equal two DataFrames"""
        return (
            self.read_data_frames()
            and self.equal_columns
            and self.equal_rows_count
            and self.equal_columns_types
            and self.equal_dataframes
        )


class PreparationDataFramesCmp:
    """Preparation DataFrame for cmp command"""

    df: DataFrame

    def __init__(self, df_path: str) -> None:
        """Initial equals DataFrames"""
        self.df_path: str = df_path

    def read_data_frame(self) -> bool:
        """Read pickle file"""
        if getattr(self, "df", None) is None:
            try:
                self.df = pd.read_pickle(self.df_path)
            except (FileNotFoundError, OSError, UnpicklingError):
                return False
        return bool(len(self.df))

    def get_types(self) -> Dict[str, object]:
        """Equals columns types"""
        types: Dict[str, str] = dict(self.df.dtypes)
        if len(self.df):

            dtypes: Dict[str, object] = {
                "int64": int,
                "str": str,
                "datetime64[ns]": datetime.datetime,
            }
            for col, tp in types.items():
                if str(tp) != "object":
                    types[col] = dtypes.get(types[col], "object")
                else:
                    first_index_row: int = self.df[col].first_valid_index()
                    if first_index_row is not None:
                        types[col] = type(self.df[col].loc[first_index_row])
        return types

    def order_df(self) -> None:
        """Order columns in DataFrame"""
        for col in (col for col, tp in self.get_types().items() if tp == list):
            self.df[col] = self.df[col].apply(
                lambda x: x.sort() or tuple(x) if x else None
            )
        self.df.sort_values(by=self.df.columns.to_list(), inplace=True, ignore_index=True)

    def save_df(self, file_name: str) -> None:
        """Save temp file to equals"""
        self.df.to_csv(f"./{file_name}.csv", index=False)


# Сравнение двух датасетов способом 1
def equals_two_df(df_paths: List[str]) -> bool:
    """Function for equal two DataFrames"""
    equals = EqualDataFrames(*df_paths)
    return equals.equals


# Сравнение двух датасетов способом 2
def equals_two_df_cmp(df_paths: List[str]) -> bool:
    """Function for equal two DataFrames"""
    for df_path in df_paths:
        prepare = PreparationDataFramesCmp(df_path=df_path)
        if not prepare.read_data_frame():
            return False
        prepare.order_df()
        prepare.save_df(df_path.replace(".pkl", "_temp"))
        del prepare
    try:
        subprocess.check_output(
            f"cmp {' '.join((df_path.replace('.pkl', '_temp') + '.csv' for df_path in df_paths))}",
            shell=True,
        )
    except subprocess.CalledProcessError:
        return False
    else:
        return True
    finally:
        pass
        # Remove temp files
        for df_path in df_paths:
            if os.path.exists(f"{df_path.replace('.pkl', '_temp')}.csv"):
                os.remove(f"{df_path.replace('.pkl', '_temp')}.csv")


if __name__ == "__main__":
    data_generator = DataGenerator(file_size=200 * 1024 * 1024 * 1024, columns_count=4)
    data_generator.dtypes = {'a': int, 'b': str, 'c': datetime.datetime, 'd': list}
    # data_generator.generate_df()

    equal_data_generator = data_generator.create_copy()
    equal_data_generator.save_df()
    del equal_data_generator

    shuffle_rows_data_generator = data_generator.create_copy()
    shuffle_rows_data_generator.shuffle_rows()
    shuffle_rows_data_generator.file_name = "shuffle_rows_test_data"
    shuffle_rows_data_generator.save_df()
    del shuffle_rows_data_generator

    shuffle_columns_data_generator = data_generator.create_copy()
    shuffle_columns_data_generator.shuffle_columns()
    shuffle_columns_data_generator.file_name = "shuffle_columns_test_data"
    shuffle_columns_data_generator.save_df()
    del shuffle_columns_data_generator

    add_column_data_generator = data_generator.create_copy()
    add_column_data_generator.add_column()
    add_column_data_generator.file_name = "add_columns_test_data"
    add_column_data_generator.save_df()
    del add_column_data_generator

    change_data_generator = data_generator.create_copy()
    change_data_generator.change_data_in_column(change_count=100)
    change_data_generator.file_name = "change_test_data"
    change_data_generator.save_df()
    del change_data_generator

    change_rows_data_generator = data_generator.create_copy()
    change_rows_data_generator.drop_rows()
    change_rows_data_generator.file_name = "change_rows_test_data"
    change_rows_data_generator.save_df()
    del change_rows_data_generator

    del data_generator

    gc.collect()

    # Способ #1 (не подходит - требует много памяти)
    print("copy_test_data:", equals_two_df(["test_data.pkl", "copy_test_data.pkl"]))
    print("add_columns_test_data:", equals_two_df(["test_data.pkl", "add_columns_test_data.pkl"]))
    print("shuffle_columns_test_data:", equals_two_df(["test_data.pkl", "shuffle_columns_test_data.pkl"]))
    print("shuffle_rows_test_data:", equals_two_df(["test_data.pkl", "shuffle_rows_test_data.pkl"]))
    print("change_test_data:", equals_two_df(["test_data.pkl", "change_test_data.pkl"]))
    print("change_rows_test_data:", equals_two_df(["test_data.pkl", "change_rows_test_data.pkl"]))

    print('----------------------------')
    # Способ #2
    print("copy_test_data:", equals_two_df_cmp(["test_data.pkl", "copy_test_data.pkl"]))
    print("add_columns_test_data:", equals_two_df_cmp(["test_data.pkl", "add_columns_test_data.pkl"]))
    print("shuffle_columns_test_data:", equals_two_df_cmp(["test_data.pkl", "shuffle_columns_test_data.pkl"]))
    print("shuffle_rows_test_data:", equals_two_df_cmp(["test_data.pkl", "shuffle_rows_test_data.pkl"]))
    print("change_test_data:", equals_two_df_cmp(["test_data.pkl", "change_test_data.pkl"]))
    print("change_rows_data:", equals_two_df_cmp(["test_data.pkl", "change_rows_data.pkl"]))
